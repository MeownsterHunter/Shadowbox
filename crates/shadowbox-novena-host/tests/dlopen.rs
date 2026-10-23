use libloading::{Library, Symbol};
use std::ffi::{c_char, c_void, CStr, CString};
use std::fs;
use std::path::PathBuf;
use std::process::Command;

#[repr(C)]
#[derive(Default)]
struct Registers {
    x: [u64; 8],
    d: [u64; 8],
    sp: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Host {
    user: *mut c_void,
    read_memory: Option<unsafe extern "C" fn(*mut c_void, u64, *mut u8, u64) -> i32>,
    write_memory: Option<unsafe extern "C" fn(*mut c_void, u64, *const u8, u64) -> i32>,
    present: Option<unsafe extern "C" fn(*mut c_void, u64, u32, u32, *const u8, u64)>,
    render_scale: f32,
    wait_vblank: Option<unsafe extern "C" fn(*mut c_void)>,
}

unsafe extern "C" fn read_memory(user: *mut c_void, address: u64, out: *mut u8, size: u64) -> i32 {
    let memory = unsafe { &*(user.cast::<Vec<u8>>()) };
    let start = address.saturating_sub(0x1000) as usize;
    let end = start.saturating_add(size as usize);
    if end > memory.len() {
        return 1;
    }
    unsafe { std::slice::from_raw_parts_mut(out, size as usize) }
        .copy_from_slice(&memory[start..end]);
    0
}

#[test]
fn loaded_cdylib_translates_fixture_and_writes_clean_census() {
    let target_dir = std::env::var_os("CARGO_TARGET_DIR")
        .map(|path| PathBuf::from(path).join("debug"))
        .or_else(|| {
            std::env::current_exe().ok().and_then(|path| {
                path.parent()
                    .and_then(|deps| deps.parent())
                    .map(PathBuf::from)
            })
        })
        .expect("could not determine Cargo target directory");
    let library = target_dir.join(format!(
        "{}shadowbox_novena_host{}",
        std::env::consts::DLL_PREFIX,
        std::env::consts::DLL_SUFFIX
    ));
    if !library.exists() {
        let workspace = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        assert!(
            Command::new("cargo")
                .current_dir(workspace)
                .args(["build", "-p", "shadowbox-novena-host"])
                .status()
                .expect("failed to build shadowbox-novena-host cdylib")
                .success(),
            "building shadowbox-novena-host cdylib failed"
        );
    }
    let corpus = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../corpus");
    let fixture = fs::read(corpus.join("constant_color.bin")).unwrap();
    let header = fs::read(corpus.join("constant_color.hdr")).unwrap();
    let mut memory = vec![0; 0x4000];
    // Graphics program layout magic (novena docs/provenance program layout).
    memory[0..4].copy_from_slice(&0x1234_5678u32.to_le_bytes());
    // SM 5.x shader headers are 80 bytes; the corpus sidecar is padded.
    memory[0x1030 - 0x1000..0x1080 - 0x1000].copy_from_slice(&header[..0x50]);
    memory[0x1080 - 0x1000..0x1080 - 0x1000 + fixture.len()].copy_from_slice(&fixture);
    // The guest compiler ends a program with a bundle holding an unconditional
    // branch to itself; novena reads code up to it (its provenance 0046).
    // Scheduling word, two NOPs, BRA-to-self (novena tests/shadowbox/drawing.rs).
    let end = 0x1080 - 0x1000 + fixture.len();
    for (slot, word) in [
        0,
        0x50b0_0000_0007_0f00,
        0x50b0_0000_0007_0f00,
        0xe240_0fff_ff87_000f_u64,
    ]
    .into_iter()
    .enumerate()
    {
        memory[end + 8 * slot..end + 8 * slot + 8].copy_from_slice(&word.to_le_bytes());
    }
    let mut record = [0u8; 0x40];
    // Records point at a 0x30-byte prefix; the 80-byte header follows and code starts at +0x80.
    record[..8].copy_from_slice(&0x1000u64.to_le_bytes());
    memory[0x2000 - 0x1000..0x2040 - 0x1000].copy_from_slice(&record);
    let mut memory = Box::new(memory);
    let host = Host {
        user: (&mut *memory as *mut Vec<u8>).cast(),
        read_memory: Some(read_memory),
        write_memory: None,
        present: None,
        render_scale: 1.0,
        wait_vblank: None,
    };
    unsafe {
        let library = Library::new(library).unwrap();
        let version: Symbol<unsafe extern "C" fn() -> *const c_char> =
            library.get(b"novena_version\0").unwrap();
        let host_version: Symbol<unsafe extern "C" fn() -> u32> =
            library.get(b"novena_host_interface_version\0").unwrap();
        assert_eq!(CStr::from_ptr(version()).to_str().unwrap(), "0.0.1");
        assert_eq!(host_version(), novena::HOST_INTERFACE_VERSION);
        let create: Symbol<unsafe extern "C" fn(*const Host) -> *mut novena::Instance> =
            library.get(b"novena_instance_create\0").unwrap();
        let request: Symbol<unsafe extern "C" fn(*const novena::Instance, *const c_char) -> u32> =
            library.get(b"novena_instance_request\0").unwrap();
        let call: Symbol<
            unsafe extern "C" fn(*const novena::Instance, u32, *mut Registers) -> i32,
        > = library.get(b"novena_instance_call\0").unwrap();
        let census: Symbol<unsafe extern "C" fn(*const novena::Instance, *const c_char) -> i32> =
            library.get(b"novena_instance_write_census\0").unwrap();
        let destroy: Symbol<unsafe extern "C" fn(*mut novena::Instance)> =
            library.get(b"novena_instance_destroy\0").unwrap();
        let instance = create(&host);
        let init = CString::new("nvnProgramInitialize").unwrap();
        let shaders = CString::new("nvnProgramSetShaders").unwrap();
        let pool_defaults = CString::new("nvnMemoryPoolBuilderSetDefaults").unwrap();
        let pool_storage = CString::new("nvnMemoryPoolBuilderSetStorage").unwrap();
        let pool_init = CString::new("nvnMemoryPoolInitialize").unwrap();
        let init_id = request(instance, init.as_ptr());
        let shaders_id = request(instance, shaders.as_ptr());
        let pool_defaults_id = request(instance, pool_defaults.as_ptr());
        let pool_storage_id = request(instance, pool_storage.as_ptr());
        let pool_init_id = request(instance, pool_init.as_ptr());
        let mut registers = Registers {
            x: [7, 0, 0, 0, 0, 0, 0, 0],
            ..Default::default()
        };
        assert_eq!(call(instance, pool_defaults_id, &mut registers), 0);
        registers.x = [7, 0x1000, 0x3000, 0, 0, 0, 0, 0];
        assert_eq!(call(instance, pool_storage_id, &mut registers), 0);
        registers.x = [8, 7, 0, 0, 0, 0, 0, 0];
        assert_eq!(call(instance, pool_init_id, &mut registers), 0);
        registers.x = [9, 1, 0, 0, 0, 0, 0, 0];
        assert_eq!(call(instance, init_id, &mut registers), 0);
        registers.x = [9, 1, 0x2000, 0, 0, 0, 0, 0];
        assert_eq!(call(instance, shaders_id, &mut registers), 0);
        let path =
            std::env::temp_dir().join(format!("shadowbox-host-census-{}", std::process::id()));
        let path_c = CString::new(path.to_str().unwrap()).unwrap();
        assert_eq!(census(instance, path_c.as_ptr()), 0);
        let report = fs::read_to_string(&path).unwrap();
        assert!(report.contains("1 yes nvnProgramSetShaders"));
        assert!(!report.contains("shader-translation-error"), "{report}");
        fs::remove_file(path).unwrap();
        destroy(instance);
    }
}
