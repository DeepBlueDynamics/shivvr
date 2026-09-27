/// Build script: generate empty stub .lib files for DirectX/DirectML libraries
/// that ort-sys requires at link time but aren't present in older Windows SDKs.
/// CPU-only inference works fine — these are only needed if DirectML GPU is used.

fn main() {
    #[cfg(target_os = "windows")]
    {
        let out_dir = std::env::var("OUT_DIR").unwrap();
        let stubs = ["DXCORE", "DirectML", "D3D12"];

        // Find lib.exe from Visual Studio
        let lib_exe = find_lib_exe();

        for name in &stubs {
            let lib_path = format!("{}\\{}.lib", out_dir, name);
            

            if let Some(ref lib_exe) = lib_exe {
                // Use lib.exe to create a proper empty import library
                let def_path = format!("{}\\{}.def", out_dir, name);
                std::fs::write(&def_path, format!("LIBRARY {}\nEXPORTS\n", name)).ok();
                let status = std::process::Command::new(lib_exe)
                    .args(&[
                        "/NOLOGO",
                        &format!("/DEF:{}", def_path),
                        &format!("/OUT:{}", lib_path),
                        "/MACHINE:X64",
                    ])
                    .status();
                if let Ok(s) = status {
                    if s.success() {
                        continue;
                    }
                }
            }

            // Fallback: create minimal COFF archive
            // MSVC lib.exe output format: signature + two linker members + longnames
            let lib = create_empty_coff_lib();
            std::fs::write(&lib_path, &lib).unwrap_or_else(|e| {
                println!("cargo:warning=Failed to create stub {}.lib: {}", name, e);
            });
        }

        println!("cargo:rustc-link-search=native={}", out_dir);
    }
}

#[cfg(target_os = "windows")]
fn find_lib_exe() -> Option<String> {
    // 1. Explicit override: LIB_EXE=<full path to lib.exe>.
    println!("cargo:rerun-if-env-changed=LIB_EXE");
    if let Ok(path) = std::env::var("LIB_EXE") {
        if std::path::Path::new(&path).exists() {
            return Some(path);
        }
        println!("cargo:warning=LIB_EXE is set but does not exist: {}", path);
    }
    // 2. Ask the Visual Studio installer where the newest MSVC toolchain is.
    if let Some(path) = vswhere_lib_exe() {
        return Some(path);
    }
    // 3. Anything already on PATH (e.g. inside a Developer Command Prompt).
    which_lib_exe()
}

/// Locate lib.exe through vswhere.exe, which ships with every Visual Studio
/// and Build Tools install since 2017. Works for any edition and MSVC version,
/// unlike a hardcoded path.
#[cfg(target_os = "windows")]
fn vswhere_lib_exe() -> Option<String> {
    let program_files = std::env::var("ProgramFiles(x86)")
        .or_else(|_| std::env::var("ProgramFiles"))
        .ok()?;
    let vswhere = std::path::Path::new(&program_files)
        .join("Microsoft Visual Studio")
        .join("Installer")
        .join("vswhere.exe");
    if !vswhere.exists() {
        return None;
    }
    let output = std::process::Command::new(vswhere)
        .args([
            "-latest",
            "-products",
            "*",
            "-requires",
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "-find",
            r"VC\Tools\MSVC\**\bin\Hostx64\x64\lib.exe",
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    // Several MSVC versions may be installed side by side; vswhere lists them
    // in ascending order, so the last line is the newest.
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .last()
        .map(str::to_string)
}

#[cfg(target_os = "windows")]
fn which_lib_exe() -> Option<String> {
    let output = std::process::Command::new("where").arg("lib.exe").output().ok()?;
    if output.status.success() {
        let path = String::from_utf8_lossy(&output.stdout);
        path.lines().next().map(|s| s.trim().to_string())
    } else {
        None
    }
}

#[cfg(target_os = "windows")]
fn create_empty_coff_lib() -> Vec<u8> {
    // Create a minimal valid COFF archive with two linker members
    let mut lib = Vec::new();

    // Archive signature (8 bytes)
    lib.extend_from_slice(b"!<arch>\n");

    // First linker member: "/" with 4 bytes content (0 symbols)
    write_member_header(&mut lib, b"/               ", 4);
    lib.extend_from_slice(&0u32.to_be_bytes()); // 0 symbols (big-endian)

    // Second linker member: "/" with 8 bytes content (0 members, 0 symbols)
    write_member_header(&mut lib, b"/               ", 8);
    lib.extend_from_slice(&0u32.to_le_bytes()); // 0 members (little-endian)
    lib.extend_from_slice(&0u32.to_le_bytes()); // 0 symbols

    lib
}

#[cfg(target_os = "windows")]
fn write_member_header(buf: &mut Vec<u8>, name: &[u8; 16], size: u32) {
    buf.extend_from_slice(name);                            // name: 16 bytes
    buf.extend_from_slice(b"0           ");                 // date: 12 bytes
    buf.extend_from_slice(b"0     ");                       // uid:  6 bytes
    buf.extend_from_slice(b"0     ");                       // gid:  6 bytes
    buf.extend_from_slice(b"0       ");                     // mode: 8 bytes
    buf.extend_from_slice(format!("{:<10}", size).as_bytes()); // size: 10 bytes
    buf.extend_from_slice(b"`\n");                          // end:  2 bytes
}
