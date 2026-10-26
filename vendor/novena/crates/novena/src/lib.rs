//! A graphics-call library with CPU execution and an optional Vulkan backend.
//!
//! A host creates an [`Instance`], owns program memory, and forwards calls with
//! [`Registers`]. The library records objects and command lists, counts calls,
//! and executes supported clears, copies, and presentation. Unsupported calls
//! return [`Status::Unimplemented`].
//!
//! The `vulkan` feature adds arena-backed resources and bounded graphics draws.
//! Draw execution needs explicit host contracts for unresolved formats and state.
//! [`ShaderTranslator`] supplies translated SPIR-V; no translator is included.
//! [`startup_cache`] loads translated output and schedules owned worker requests.
//!
//! Guest behavior is limited to recorded signature and shape observations.
//! Public host execution uses
//! Vulkan contracts and original synthetic tests.
#![deny(missing_docs)]

pub mod api;
pub mod counters;
#[cfg(any(test, feature = "vulkan"))]
mod dirty_ranges;
#[cfg(feature = "draw-metrics")]
pub mod draw_metrics;
mod execution;
pub mod functions;
pub mod global_memory;
#[cfg(feature = "vulkan")]
pub mod gpu;
mod instance;
pub mod observe;
pub mod presentation;
pub use presentation::{FrameStatistics, PresentationConfig, PresentationMode};
pub mod startup_cache;
pub mod surfaces;
pub mod tiling;
mod unsupported;
mod workers;

pub use instance::{
    Census, Host, HostVulkan, Instance, Registers, ShaderStage, ShaderTranslator, Status,
    SHADER_STAGE_UNKNOWN,
};

use functions::FunctionId;
use std::cell::RefCell;
use std::ffi::{c_char, CStr, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::Path;

/// Version of the host interface. It changes when a host built against an
/// older `include/novena.h` could no longer use the library.
pub const HOST_INTERFACE_VERSION: u32 = 5;

/// Returned by lookups for a name the library does not know.
pub const FUNCTION_NONE: u32 = u32::MAX;

static VERSION: &str = concat!(env!("CARGO_PKG_VERSION"), "\0");

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

fn ffi_guard<T>(default: T, f: impl FnOnce() -> T) -> T {
    LAST_ERROR.with(|error| *error.borrow_mut() = None);
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(value) => value,
        Err(payload) => {
            let message = payload
                .downcast_ref::<&str>()
                .copied()
                .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
                .unwrap_or("unknown panic");
            LAST_ERROR.with(|error| {
                *error.borrow_mut() = Some(
                    CString::new(message)
                        .unwrap_or_else(|_| CString::new("panic message contained NUL").unwrap()),
                )
            });
            default
        }
    }
}

/// The most recent panic message on this thread, or null when there was none.
#[no_mangle]
pub extern "C" fn novena_last_error() -> *const c_char {
    // Not wrapped in `ffi_guard`, which would clear the message first.
    LAST_ERROR.with(|error| {
        error
            .borrow()
            .as_ref()
            .map_or(std::ptr::null(), |message| message.as_ptr())
    })
}

/// The library version as a NUL-terminated string that lives as long as the
/// library is loaded.
#[no_mangle]
pub extern "C" fn novena_version() -> *const c_char {
    ffi_guard(std::ptr::null(), || VERSION.as_ptr().cast())
}

/// The host interface version this library was built with.
#[no_mangle]
pub extern "C" fn novena_host_interface_version() -> u32 {
    ffi_guard(0, || HOST_INTERFACE_VERSION)
}

/// Number of functions in the table. Valid function ids are `0..count`.
#[no_mangle]
pub extern "C" fn novena_function_count() -> u32 {
    ffi_guard(0, || functions::count() as u32)
}

/// Name of a function as a NUL-terminated string owned by the library, or
/// null for an id outside the table.
#[no_mangle]
pub extern "C" fn novena_function_name(function: u32) -> *const c_char {
    ffi_guard(std::ptr::null(), || {
        functions::c_name(FunctionId(function)).map_or(std::ptr::null(), CStr::as_ptr)
    })
}

/// Id of the function with this name, or `FUNCTION_NONE`.
///
/// # Safety
/// `name` must be null or point to a NUL-terminated string.
#[no_mangle]
pub unsafe extern "C" fn novena_function_lookup(name: *const c_char) -> u32 {
    ffi_guard(FUNCTION_NONE, || {
        if name.is_null() {
            return FUNCTION_NONE;
        }
        // SAFETY: the caller passes a NUL-terminated string.
        let name = unsafe { CStr::from_ptr(name) };
        name.to_str()
            .ok()
            .and_then(functions::lookup)
            .map_or(FUNCTION_NONE, |id| id.0)
    })
}

/// Enable host write notifications for a registered pool and its aliases.
/// Every subsequent host storage write must be reported before submission.
///
/// # Safety
/// `instance` must be a live instance pointer, or null.
#[no_mangle]
pub unsafe extern "C" fn novena_instance_track_pool_writes(
    instance: *const Instance,
    pool: u64,
) -> Status {
    ffi_guard(Status::InternalError, || {
        let Some(instance) = (unsafe { instance.as_ref() }) else {
            return Status::BadArgument;
        };
        #[cfg(feature = "vulkan")]
        {
            if instance.track_pool_writes(pool) {
                Status::Ok
            } else {
                Status::BadArgument
            }
        }
        #[cfg(not(feature = "vulkan"))]
        {
            let _ = (instance, pool);
            Status::Unimplemented
        }
    })
}

/// Report a host storage write by byte address and length before the next submission.
///
/// # Safety
/// `instance` must be a live instance pointer, or null.
#[no_mangle]
pub unsafe extern "C" fn novena_instance_notify_memory_write(
    instance: *const Instance,
    address: u64,
    size: u64,
) -> Status {
    ffi_guard(Status::InternalError, || {
        let Some(instance) = (unsafe { instance.as_ref() }) else {
            return Status::BadArgument;
        };
        #[cfg(feature = "vulkan")]
        {
            if instance.notify_memory_write(address, size) {
                Status::Ok
            } else {
                Status::BadArgument
            }
        }
        #[cfg(not(feature = "vulkan"))]
        {
            let _ = (instance, address, size);
            Status::Unimplemented
        }
    })
}

/// Adopt coherent arena storage for a registered pool. Returns null on failure.
///
/// # Safety
/// `instance` must be live or null. The host must satisfy the mapping and access
/// contract of `GlobalMemory::map_pool`, including callback routing and waits.
#[no_mangle]
pub unsafe extern "C" fn novena_instance_map_pool(instance: *const Instance, pool: u64) -> *mut u8 {
    ffi_guard(std::ptr::null_mut(), || {
        let Some(instance) = (unsafe { instance.as_ref() }) else {
            return std::ptr::null_mut();
        };
        #[cfg(feature = "vulkan")]
        {
            unsafe { instance.map_pool(pool) }.unwrap_or(std::ptr::null_mut())
        }
        #[cfg(not(feature = "vulkan"))]
        {
            let _ = (instance, pool);
            std::ptr::null_mut()
        }
    })
}

/// Wait for device access to a mapped host range. Nonzero `write` waits for readers too.
///
/// # Safety
/// `instance` must be a live instance pointer, or null. Serialize access to mapped
/// storage with submissions until the host read or write has finished.
#[no_mangle]
pub unsafe extern "C" fn novena_instance_wait_pool(
    instance: *const Instance,
    pool: u64,
    offset: u64,
    size: u64,
    write: u32,
) -> Status {
    ffi_guard(Status::InternalError, || {
        let Some(instance) = (unsafe { instance.as_ref() }) else {
            return Status::BadArgument;
        };
        let Ok(size) = usize::try_from(size) else {
            return Status::BadArgument;
        };
        #[cfg(feature = "vulkan")]
        {
            if instance.wait_pool(pool, offset, size, write != 0) {
                Status::Ok
            } else {
                Status::BadArgument
            }
        }
        #[cfg(not(feature = "vulkan"))]
        {
            let _ = (instance, pool, offset, size, write);
            Status::Unimplemented
        }
    })
}

/// Create an instance. `host` may be null for a host that provides nothing
/// yet; the structure is copied. Destroy the result with
/// `novena_instance_destroy`.
///
/// # Safety
/// `host` must be null or point to a valid `Host`. Its `user` pointer and
/// callbacks must stay usable for the life of the instance and must tolerate
/// being used from several threads at once.
#[no_mangle]
pub unsafe extern "C" fn novena_instance_create(host: *const Host) -> *mut Instance {
    ffi_guard(std::ptr::null_mut(), || {
        // SAFETY: the caller passes null or a valid `Host`.
        let instance = match unsafe { host.as_ref() } {
            Some(host) => unsafe { Instance::with_host(*host) },
            None => Instance::new(),
        };
        Box::into_raw(Box::new(instance))
    })
}

/// Destroy an instance. Null is accepted.
///
/// # Safety
/// `instance` must be null or come from `novena_instance_create` and not
/// have been destroyed already. No other call on the instance may be running
/// or start afterwards.
#[no_mangle]
pub unsafe extern "C" fn novena_instance_destroy(instance: *mut Instance) {
    ffi_guard((), || {
        if !instance.is_null() {
            drop(unsafe { Box::from_raw(instance) });
        }
    })
}

/// Set presentation capacity and mode, zero for FIFO or one for mailbox.
///
/// # Safety
/// `instance` must be live. Presentation callbacks must not reenter the library.
#[no_mangle]
pub unsafe extern "C" fn novena_instance_set_presentation(
    instance: *const Instance,
    frames_in_flight: u32,
    mode: u32,
) -> Status {
    ffi_guard(Status::InternalError, || {
        let Some(instance) = (unsafe { instance.as_ref() }) else {
            return Status::BadArgument;
        };
        let mode = match mode {
            0 => PresentationMode::Fifo,
            1 => PresentationMode::Mailbox,
            _ => return Status::BadArgument,
        };
        instance.configure_presentation(PresentationConfig {
            frames_in_flight,
            mode,
        })
    })
}

/// Deliver one ready callback, or drain all pending callbacks when wait is one.
///
/// # Safety
/// `instance` must be live. Presentation callbacks must not reenter the library.
#[no_mangle]
pub unsafe extern "C" fn novena_instance_poll_presentations(
    instance: *const Instance,
    wait: u32,
) -> Status {
    ffi_guard(Status::InternalError, || {
        let Some(instance) = (unsafe { instance.as_ref() }) else {
            return Status::BadArgument;
        };
        if wait > 1 {
            return Status::BadArgument;
        }
        instance.poll_presentations(wait != 0)
    })
}

/// Copy cumulative frame counts and latency moments into host storage.
///
/// # Safety
/// `instance` must be live and `out` must be writable for one FrameStatistics.
#[no_mangle]
pub unsafe extern "C" fn novena_instance_frame_statistics(
    instance: *const Instance,
    out: *mut FrameStatistics,
) -> Status {
    ffi_guard(Status::InternalError, || {
        let Some(instance) = (unsafe { instance.as_ref() }) else {
            return Status::BadArgument;
        };
        let Some(out) = (unsafe { out.as_mut() }) else {
            return Status::BadArgument;
        };
        *out = instance.frame_statistics();
        Status::Ok
    })
}

/// Set or clear the directory for local debugging dumps of translated
/// shaders. The files contain translated output derived from the observed
/// program's shaders and never the original shader bytes.
///
/// # Safety
/// `instance` must be live and `directory` must be null or NUL-terminated
/// UTF-8.
#[no_mangle]
pub unsafe extern "C" fn novena_instance_set_shader_dump_directory(
    instance: *const Instance,
    directory: *const c_char,
) -> Status {
    ffi_guard(Status::InternalError, || {
        let Some(instance) = (unsafe { instance.as_ref() }) else {
            return Status::BadArgument;
        };
        let path = if directory.is_null() {
            None
        } else {
            let value = unsafe { CStr::from_ptr(directory) };
            let Ok(value) = value.to_str() else {
                return Status::BadArgument;
            };
            Some(Path::new(value))
        };
        instance.set_shader_dump_directory(path);
        Status::Ok
    })
}

/// The program asked its bootstrap function for `name`. Records the request
/// and returns the function id, or `FUNCTION_NONE` for a name the library
/// does not know (which is recorded too).
///
/// # Safety
/// `instance` must be live; `name` must be null or NUL-terminated.
#[no_mangle]
pub unsafe extern "C" fn novena_instance_request(
    instance: *const Instance,
    name: *const c_char,
) -> u32 {
    ffi_guard(FUNCTION_NONE, || {
        let Some(instance) = (unsafe { instance.as_ref() }) else {
            return FUNCTION_NONE;
        };
        if name.is_null() {
            return FUNCTION_NONE;
        }
        let name = unsafe { CStr::from_ptr(name) }.to_string_lossy();
        instance.request(&name).map_or(FUNCTION_NONE, |id| id.0)
    })
}

/// The program called `function`. `registers` holds its argument registers
/// on entry and receives its result registers.
///
/// # Safety
/// `instance` must be live. `registers` must point to a valid `Registers`
/// that nothing else reads or writes until the call returns. Calls on one
/// instance may run on several threads at once, each with its own registers.
#[no_mangle]
pub unsafe extern "C" fn novena_instance_call(
    instance: *const Instance,
    function: u32,
    registers: *mut Registers,
) -> Status {
    ffi_guard(Status::InternalError, || {
        match unsafe { (instance.as_ref(), registers.as_mut()) } {
            (Some(instance), Some(registers)) => instance.call(FunctionId(function), registers),
            _ => Status::BadArgument,
        }
    })
}

/// The original implementation of `function` returned. `registers` holds
/// its result registers. Only hosts that let the original implementation run
/// call this; it lets results be sampled along with arguments.
///
/// # Safety
/// `instance` must be live and `registers` must point to a valid `Registers`.
#[no_mangle]
pub unsafe extern "C" fn novena_instance_returned(
    instance: *const Instance,
    function: u32,
    registers: *const Registers,
) -> Status {
    ffi_guard(Status::InternalError, || {
        match unsafe { (instance.as_ref(), registers.as_ref()) } {
            (Some(instance), Some(registers)) => instance.returned(FunctionId(function), registers),
            _ => Status::BadArgument,
        }
    })
}

/// Write the sampled argument and result shapes to a text file.
///
/// # Safety
/// `instance` must be live; `path` must be NUL-terminated.
#[no_mangle]
pub unsafe extern "C" fn novena_instance_write_shapes(
    instance: *const Instance,
    path: *const c_char,
) -> Status {
    ffi_guard(Status::InternalError, || {
        // SAFETY: the caller passes a live instance.
        let Some(instance) = (unsafe { instance.as_ref() }) else {
            return Status::BadArgument;
        };
        if path.is_null() {
            return Status::BadArgument;
        }
        // SAFETY: the caller passes a NUL-terminated string.
        let path = unsafe { CStr::from_ptr(path) }.to_string_lossy();
        match std::fs::write(&*path, instance.shapes_report()) {
            Ok(()) => Status::Ok,
            Err(_) => Status::BadArgument,
        }
    })
}

/// How many times `function` was called on this instance.
///
/// # Safety
/// `instance` must be null or live.
#[no_mangle]
pub unsafe extern "C" fn novena_instance_call_count(
    instance: *const Instance,
    function: u32,
) -> u64 {
    ffi_guard(0, || {
        unsafe { instance.as_ref() }.map_or(0, |instance| instance.call_count(FunctionId(function)))
    })
}

/// Write the census (what was requested and called) to a text file.
/// Returns `Status::Ok`, or `Status::BadArgument` when the file could not be
/// written.
///
/// # Safety
/// `instance` must be live; `path` must be NUL-terminated.
#[no_mangle]
pub unsafe extern "C" fn novena_instance_write_census(
    instance: *const Instance,
    path: *const c_char,
) -> Status {
    ffi_guard(Status::InternalError, || {
        // SAFETY: the caller passes a live instance.
        let Some(instance) = (unsafe { instance.as_ref() }) else {
            return Status::BadArgument;
        };
        if path.is_null() {
            return Status::BadArgument;
        }
        // SAFETY: the caller passes a NUL-terminated string.
        let path = unsafe { CStr::from_ptr(path) }.to_string_lossy();
        match std::fs::write(&*path, instance.census().to_string()) {
            Ok(()) => Status::Ok,
            Err(_) => Status::BadArgument,
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;

    #[test]
    fn version_is_a_c_string_matching_the_package() {
        // SAFETY: `novena_version` returns a static NUL-terminated string.
        let version = unsafe { CStr::from_ptr(novena_version()) };
        assert_eq!(version.to_str(), Ok(env!("CARGO_PKG_VERSION")));
        assert_eq!(novena_host_interface_version(), HOST_INTERFACE_VERSION);
    }

    #[test]
    fn a_panic_is_reported_and_cleared_by_the_next_call() {
        let status = ffi_guard(Status::Ok, || -> Status { panic!("boom") });
        assert_eq!(status, Status::Ok);
        // SAFETY: a non-null result is a NUL-terminated string owned by the
        // library until the next call on this thread.
        let message = unsafe { CStr::from_ptr(novena_last_error()) };
        assert_eq!(message.to_str(), Ok("boom"));
        novena_version();
        assert!(novena_last_error().is_null());
    }

    #[test]
    fn c_interface_round_trips_names_and_counts_calls() {
        let count = novena_function_count();
        assert!(count > 0);
        // SAFETY: ids below the count have names; the strings are static.
        let first = unsafe { CStr::from_ptr(novena_function_name(0)) }.to_owned();
        assert_eq!(unsafe { novena_function_lookup(first.as_ptr()) }, 0);
        assert!(novena_function_name(count).is_null());

        let unknown = CString::new("notAFunction").unwrap();
        assert_eq!(
            unsafe { novena_function_lookup(unknown.as_ptr()) },
            FUNCTION_NONE
        );

        // SAFETY: a null host is allowed, and the instance is destroyed below.
        let instance = unsafe { novena_instance_create(std::ptr::null()) };
        assert_eq!(
            unsafe { novena_instance_request(instance, first.as_ptr()) },
            0
        );
        assert_eq!(
            unsafe { novena_instance_request(instance, unknown.as_ptr()) },
            FUNCTION_NONE
        );

        let mut registers = Registers::default();
        registers.x[0] = 7;
        let status = unsafe { novena_instance_call(instance, 0, &mut registers) };
        assert_eq!(status, Status::Unimplemented);
        assert_eq!(registers.x[0], 0);
        assert_eq!(unsafe { novena_instance_call_count(instance, 0) }, 1);
        assert_eq!(
            unsafe { novena_instance_call(instance, count, &mut registers) },
            Status::BadFunction
        );

        unsafe { novena_instance_destroy(instance) };
    }
}
