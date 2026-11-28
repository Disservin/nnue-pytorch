// Compile the data loader's skip filter (data_loader/cpp/training_data_loader.cpp)
// together with a small C ABI scanner, so this tool applies exactly the
// filter that training uses.
fn main() {
    let loader = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data_loader/cpp");
    let loader = loader.canonicalize().expect("data_loader/cpp not found next to tools/");

    println!("cargo:rerun-if-changed=cpp/scan.cpp");
    println!("cargo:rerun-if-changed={}", loader.join("training_data_loader.cpp").display());
    println!("cargo:rerun-if-changed={}", loader.join("training_data_loader_internal.h").display());
    println!("cargo:rerun-if-changed={}", loader.join("training_data_loader_structs.h").display());
    println!("cargo:rerun-if-changed={}", loader.join("lib").display());

    cc::Build::new()
        .cpp(true)
        .std("c++20")
        .opt_level(3)
        .define("NDEBUG", None)
        .define("HAS_BMI2", None)
        .flag("-march=native")
        .flag("-mbmi2")
        .warnings(false)
        .include(&loader)
        .include(loader.join("lib"))
        .file("cpp/scan.cpp")
        .file(loader.join("training_data_loader.cpp"))
        .compile("loader_filter");
    println!("cargo:rustc-link-lib=pthread");
}
