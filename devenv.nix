{ pkgs, lib, ... }:

{
  packages = [
    pkgs.git
    pkgs.gh
    pkgs.gnumake
    pkgs.cmake
    pkgs.ninja
    pkgs.protobuf  # needed by lance-encoding build script

    # C/C++ tools
    pkgs.autoconf
    pkgs.automake
    pkgs.pkg-config
    pkgs.clang-tools

    # Rust toolchain
    pkgs.rustup
  ];

  env.DEVELOPER_DIR = lib.mkForce "/Applications/Xcode.app/Contents/Developer";

  enterShell = ''
    export GEN=ninja
    export DEVELOPER_DIR="/Applications/Xcode.app/Contents/Developer"
    unset SDKROOT
    unset NIX_CFLAGS_COMPILE
    unset NIX_LDFLAGS
    export CC=/usr/bin/clang
    export CXX=/usr/bin/clang++
  '';

  git-hooks.hooks = {
    ripsecrets.enable = true;
    clang-format = {
      enable = true;
      types_or = [ "c++" "c" ];
    };
  };
}
