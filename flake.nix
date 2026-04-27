{
  description = "Nix Flake Dev Shell for CS 5782 Final Project";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    systems.url = "github:nix-systems/default";
  };

  outputs =
    inputs:
    let
      inherit (inputs.nixpkgs) lib;
      forAllSystems = lib.genAttrs (import inputs.systems);
      mkPkgsForSystems =
        nixpkgs:
        forAllSystems (
          system:
          import nixpkgs {
            inherit system;
            config.allowUnfree = true;
          }
        );
      pkgsBySystem = mkPkgsForSystems inputs.nixpkgs;
    in
    {
      devShells = forAllSystems (
        system:
        let
          pkgs = pkgsBySystem.${system};
        in
        {
          default = pkgs.mkShell {
            packages = [
              pkgs.uv

              pkgs.prek

              pkgs.texliveFull

              pkgs.bashInteractive
            ];
          };
        }
      );
    };
}
