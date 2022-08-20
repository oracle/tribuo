{
  description = "A very basic flake";

  outputs = { self, nixpkgs, nixpkgs-unstable, flake-utils, mvn2nix }:
    flake-utils.lib.eachDefaultSystem( system: 
      let
        pkgs = nixpkgs.legacyPackages.${system};
        unstable = nixpkgs.legacyPackages.${system};
        m2n = mvn2nix.legacyPackages.${system};
        libgomp1 = pkgs.runCommand "libgomp1" {} ''
          mkdir -p $out/lib
          ln -s ${pkgs.llvmPackages.openmp}/lib/libgomp.so $out/lib/libgomp.so.1
        '';
        buildLockFile = pkgs.writeShellScriptBin "tribuo-lock" ''
           LD_LIBRARY_PATH=${libgomp1}/lib
           ${m2n.mvn2nix}/bin/mvn2nix --verbose pom.xml > mvn2nix-lock.json
        '';
        intellij = unstable.jetbrains.idea-community;
        tribuo = pkgs.callPackage ./default.nix {
          buildMavenRepositoryFromLockFile = m2n.buildMavenRepositoryFromLockFile;
          inherit libgomp1;
        };
      in {
        packages = {
          inherit intellij buildLockFile tribuo;
        };
        devShell = pkgs.mkShell {
          LD_LIBRARY_PATH = libgomp1 + "/lib";
          buildInputs = [ pkgs.jupyter pkgs.jdk pkgs.maven3];
        };
    });
}
