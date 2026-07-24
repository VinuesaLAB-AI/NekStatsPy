"""Command-line interface: ``nekstat <post|mesh|plot|validate>``."""

import os
import argparse

import numpy as np


def _cmd_post(args):
    from .pipeline import run
    res = run(args.nek, args.mesh, args.out, args.istart, args.iend, args.long)
    cl = res.scal("ss", "Cl")[0]; cd = res.scal("ss", "Cd")[0]
    print(f"[post] wrote {args.out}")
    print(f"       {res.n} profiles/side; Cl={cl:.5f} Cd={cd:.5f} L/D={cl/cd:.2f}")


def _cmd_mesh(args):
    from .naca_geom import build_mesh, save_inter_mesh
    import scipy.io as sio
    x_pts, y_pts, npoints, yn, alphau, alphal, xc = build_mesh(tol=args.tol)
    print(f"[mesh] {npoints} points, {len(xc)} stations x {len(yn)} wall-normal")
    if args.ref:
        r = sio.loadmat(args.ref)
        m = r["x_pts"].ravel() != -1
        print(f"       max|dx vs ref|={np.max(np.abs(x_pts[m]-r['x_pts'].ravel()[m])):.2e}"
              f"  max|dalphau|={np.max(np.abs(alphau-r['alphau'].ravel())):.2e}")
    if args.write_into:
        os.makedirs(os.path.join(args.write_into, "ZSTAT"), exist_ok=True)
        save_inter_mesh(args.write_into, npoints, x_pts, y_pts)
        print(f"       wrote {args.write_into}/ZSTAT/x.fort,y.fort")


def _cmd_plot(args):
    from .pipeline import load_result
    from . import viz
    res = load_result(args.mat)
    out = viz.plot_single(res, args.figdir)
    print("[plot] wrote:", *out, sep="\n  ")


def _cmd_validate(args):
    from .pipeline import load_result
    from . import viz
    ref = load_result(args.ref)
    mine = load_result(args.mine)
    print("Max |mine - ref| over exported suction-side profiles:")
    for f in ["U", "V", "uu", "Cf", "beta", "Reth", "Ret", "H12",
              "Pkp", "Dkp", "Tkp", "VDkp", "VPGkp", "Ckp"]:
        d = max(np.max(np.abs(ref.prof("ss", p, f) - mine.prof("ss", p, f)))
                for p in range(ref.n))
        print(f"   {f:7s} {d:.3e}")
    out = viz.plot_overlay(ref, mine, args.figdir)
    print("[validate] wrote:", *out, sep="\n  ")


def _default(name):
    try:
        from ._data import sample_paths
        return sample_paths().get(name)
    except Exception:
        return None


def main(argv=None):
    p = argparse.ArgumentParser(prog="nekstat",
                                description="Nek5000 wing statistics post-processing")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("post", help="process int_fld -> stat .mat")
    sp.add_argument("--nek", default=_default("nek_dir"),
                    help="dir containing ZSTAT/int_fld")
    sp.add_argument("--mesh", default=_default("mesh"), help="int_mesh.mat")
    sp.add_argument("--out", default="stat_naca4412_200k.mat")
    sp.add_argument("--istart", type=int, default=25)
    sp.add_argument("--iend", type=int, default=80)
    sp.add_argument("--long", type=int, default=1)
    sp.set_defaults(func=_cmd_post)

    sm = sub.add_parser("mesh", help="generate x.fort/y.fort interpolation mesh")
    sm.add_argument("--tol", type=float, default=0.5e-7)
    sm.add_argument("--ref", default=_default("mesh"), help="int_mesh.mat to validate against")
    sm.add_argument("--write-into", default=None, help="dir with ZSTAT/ to write into")
    sm.set_defaults(func=_cmd_mesh)

    spl = sub.add_parser("plot", help="plot one processed .mat")
    spl.add_argument("--mat", required=True)
    spl.add_argument("--figdir", default="figs")
    spl.set_defaults(func=_cmd_plot)

    sv = sub.add_parser("validate", help="overlay + diff two processed .mat files")
    sv.add_argument("--mine", required=True)
    sv.add_argument("--ref", default=_default("reference"))
    sv.add_argument("--figdir", default="validation_figs")
    sv.set_defaults(func=_cmd_validate)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
