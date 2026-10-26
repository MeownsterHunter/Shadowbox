fn main() {
    // novena's rlib already contains its no_mangle C exports. The host
    // intentionally replaces only instance creation, so let the wrapper
    // definition win over that one duplicate at final link time.
    println!("cargo:rustc-link-arg=-Wl,--allow-multiple-definition");
}
