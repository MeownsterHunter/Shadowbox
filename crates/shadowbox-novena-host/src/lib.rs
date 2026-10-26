//! A drop-in novena cdylib with Shadowbox shader translation installed.
//!
//! Sources: novena's public C exports and instance setup are read-only
//! integration facts from `vendor/novena/crates/novena/src/lib.rs`
//! and `instance.rs`; translation behavior is delegated to this workspace's
//! `shadowbox-novena` crate.

use novena::ShaderTranslator;
use shadowbox_novena::{Dumps, Forgets, ShadowboxTranslator};
use std::ffi::c_void;
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

pub use novena::{
    novena_function_count, novena_function_lookup, novena_function_name, novena_instance_call,
    novena_instance_call_count, novena_instance_destroy, novena_instance_request,
    novena_instance_returned, novena_instance_write_census, novena_instance_write_shapes,
    novena_last_error, novena_version,
};

/// The host interface version this library speaks for the run.
///
/// This replaces novena's own export. The run says which interface its
/// loader speaks (see [`Interface`]); a value this library cannot speak is
/// reported as 0, which no loader accepts.
#[no_mangle]
pub extern "C" fn novena_host_interface_version() -> u32 {
    match Interface::of_run() {
        Some(interface) => interface.version(),
        None => 0,
    }
}

/// The host structure layouts this library accepts: version 4, and novena's
/// own (version 5 today). Version 5 only appends the optional Vulkan presentation pointer to version 4,
/// so a version-4 host is a version-5 host without native presentation:
/// frames reach it through the RGBA `present` callback.
#[derive(Clone, Copy)]
enum Interface {
    V4,
    Own,
}

static INTERFACE: OnceLock<Option<Interface>> = OnceLock::new();

impl Interface {
    /// The run's environment is the edge: `SHADOWBOX_NOVENA_HOST_INTERFACE=4`
    /// serves loaders built against version 4 (the hosted runtime in recompiled
    /// modules), unset or novena's own number is novena's own version, and
    /// anything else is no interface this library speaks.
    fn of_run() -> Option<Self> {
        *INTERFACE.get_or_init(|| match std::env::var("SHADOWBOX_NOVENA_HOST_INTERFACE") {
            Err(std::env::VarError::NotPresent) => Some(Self::Own),
            Ok(value) => match value.parse::<u32>() {
                Ok(4) => Some(Self::V4),
                Ok(novena::HOST_INTERFACE_VERSION) => Some(Self::Own),
                _ => None,
            },
            Err(std::env::VarError::NotUnicode(_)) => None,
        })
    }

    fn version(self) -> u32 {
        match self {
            Self::V4 => 4,
            Self::Own => novena::HOST_INTERFACE_VERSION,
        }
    }

    /// Read the caller's host structure in this interface's layout.
    ///
    /// # Safety
    ///
    /// `host` is null or points to a valid host structure of this layout.
    unsafe fn host(self, host: *const c_void) -> Option<novena::Host> {
        match self {
            Self::V4 => unsafe { host.cast::<HostV4>().as_ref() }.map(|&h| h.into()),
            Self::Own => unsafe { host.cast::<novena::Host>().as_ref() }.copied(),
        }
    }
}

/// Version-4 host structure: novena's [`novena::Host`] without the trailing
/// Vulkan presentation pointer.
///
/// The loader that speaks version 4, the hosted runtime in recompiled modules,
/// copies at most one 4 KiB page per memory callback and fails any larger
/// request, while novena moves pool memory in larger chunks. Read through
/// [`PagedHost`], every transfer novena asks of it is a run of page-sized
/// callbacks.
#[repr(C)]
#[derive(Clone, Copy)]
struct HostV4 {
    user: *mut c_void,
    read_memory: Option<unsafe extern "C" fn(*mut c_void, u64, *mut u8, u64) -> i32>,
    write_memory: Option<unsafe extern "C" fn(*mut c_void, u64, *const u8, u64) -> i32>,
    present: Option<unsafe extern "C" fn(*mut c_void, u64, u32, u32, *const u8, u64)>,
    render_scale: f32,
    wait_vblank: Option<unsafe extern "C" fn(*mut c_void)>,
}

/// The most bytes a version-4 loader copies in one memory callback.
const V4_TRANSFER: usize = 4096;

/// A version-4 host as novena sees it: novena's callbacks receive this
/// value as `user` and forward to the loader's own callbacks, splitting
/// memory transfers into page-sized pieces.
struct PagedHost(HostV4);

impl From<HostV4> for novena::Host {
    fn from(host: HostV4) -> Self {
        // The instance keeps calling back until the process ends; the
        // loader creates one instance and never destroys it, so the
        // forwarding value lives as long as the process.
        let paged: &'static PagedHost = Box::leak(Box::new(PagedHost(host)));
        Self {
            user: (paged as *const PagedHost).cast_mut().cast(),
            read_memory: host.read_memory.map(|_| paged_read as _),
            write_memory: host.write_memory.map(|_| paged_write as _),
            present: host.present.map(|_| paged_present as _),
            render_scale: host.render_scale,
            wait_vblank: host.wait_vblank.map(|_| paged_wait_vblank as _),
            vulkan: std::ptr::null(),
        }
    }
}

/// # Safety
///
/// `user` is the [`PagedHost`] made at instance creation.
unsafe fn paged<'a>(user: *mut c_void) -> &'a HostV4 {
    &unsafe { &*user.cast::<PagedHost>() }.0
}

unsafe extern "C" fn paged_read(user: *mut c_void, address: u64, out: *mut u8, size: u64) -> i32 {
    let host = unsafe { paged(user) };
    let Some(read) = host.read_memory else {
        return 1;
    };
    // SAFETY: novena passes `size` writable bytes at `out`.
    let out = unsafe { std::slice::from_raw_parts_mut(out, size as usize) };
    for (index, piece) in out.chunks_mut(V4_TRANSFER).enumerate() {
        let at = address + (index * V4_TRANSFER) as u64;
        let status = unsafe { read(host.user, at, piece.as_mut_ptr(), piece.len() as u64) };
        if status != 0 {
            return status;
        }
    }
    0
}

unsafe extern "C" fn paged_write(
    user: *mut c_void,
    address: u64,
    data: *const u8,
    size: u64,
) -> i32 {
    let host = unsafe { paged(user) };
    let Some(write) = host.write_memory else {
        return 1;
    };
    // SAFETY: novena passes `size` readable bytes at `data`.
    let data = unsafe { std::slice::from_raw_parts(data, size as usize) };
    for (index, piece) in data.chunks(V4_TRANSFER).enumerate() {
        let at = address + (index * V4_TRANSFER) as u64;
        let status = unsafe { write(host.user, at, piece.as_ptr(), piece.len() as u64) };
        if status != 0 {
            return status;
        }
    }
    0
}

unsafe extern "C" fn paged_present(
    user: *mut c_void,
    window: u64,
    width: u32,
    height: u32,
    rgba: *const u8,
    stride: u64,
) {
    let host = unsafe { paged(user) };
    if let Some(present) = host.present {
        unsafe { present(host.user, window, width, height, rgba, stride) }
    }
}

unsafe extern "C" fn paged_wait_vblank(user: *mut c_void) {
    let host = unsafe { paged(user) };
    if let Some(wait) = host.wait_vblank {
        unsafe { wait(host.user) }
    }
}

/// Create a novena instance with Shadowbox translation enabled.
///
/// This has the same C ABI as novena's `novena_instance_create`, with the
/// host structure in the layout of the version this library reported. The option
/// is enabled here because novena exposes it as an instance method rather than
/// an environment variable or C host option.
///
/// # Safety
///
/// Same contract as novena's own `novena_instance_create`: `host` must be
/// null or point to a valid host structure, of the version
/// [`novena_host_interface_version`] reported, for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn novena_instance_create(host: *const c_void) -> *mut novena::Instance {
    let Some(interface) = Interface::of_run() else {
        return std::ptr::null_mut();
    };
    // SAFETY: the caller contract is novena's published C interface
    // contract. Constructing the public instance directly avoids recursively
    // resolving novena's same-named export when both crates are linked into
    // this replacement cdylib. The pointer is the edge: null asks for
    // novena's own defaults, and a non-null host is valid for the call.
    let instance = match unsafe { interface.host(host) } {
        Some(host) => unsafe { novena::Instance::with_host(host) },
        None => novena::Instance::new(),
    };
    let start = START.get_or_init(Start::read);
    instance.set_shader_translator(Some(start.translator.clone()));
    instance.set_shader_translation_enabled(true);
    (start.dumps)(&instance);
    Box::into_raw(Box::new(instance))
}

/// What the process's environment said, read once at the first instance:
/// the translator for the run, and where translated modules are dumped.
struct Start {
    translator: Arc<dyn ShaderTranslator>,
    dumps: Box<dyn Fn(&novena::Instance) + Send + Sync>,
}

static START: OnceLock<Start> = OnceLock::new();

impl Start {
    /// The host's entry is its edge: the environment says once what the run
    /// is (`SHADOWBOX_TRACE_STACK=1` traces stack proofs on stderr), what
    /// becomes of failed inputs (`SHADOWBOX_FAIL_DIR` dumps each once there),
    /// and, for local debugging, where novena writes each translated module
    /// (`SHADOWBOX_DUMP_DIR`; novena never writes the original shader bytes).
    fn read() -> Self {
        let translator = match std::env::var("SHADOWBOX_TRACE_STACK").as_deref() {
            Ok("1") => with_failures(shadowbox_novena::TracedRun),
            _ => with_failures(shadowbox_novena::QuietRun),
        };
        let dumps: Box<dyn Fn(&novena::Instance) + Send + Sync> =
            match std::env::var_os("SHADOWBOX_DUMP_DIR") {
                Some(dir) => {
                    let dir = PathBuf::from(dir);
                    Box::new(move |instance| instance.set_shader_dump_directory(Some(&dir)))
                }
                None => Box::new(|_| {}),
            };
        Self { translator, dumps }
    }
}

fn with_failures<R: shadowbox_novena::Run + 'static>(run: R) -> Arc<dyn ShaderTranslator> {
    match std::env::var_os("SHADOWBOX_FAIL_DIR") {
        Some(dir) => Arc::new(ShadowboxTranslator::on(run, Dumps::to(dir))),
        None => Arc::new(ShadowboxTranslator::on(run, Forgets)),
    }
}
