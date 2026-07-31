"""Command-line interface: ``nekstat <build|run|post|mesh|plot|validate>``."""

import os
import argparse

import numpy as np


def _cmd_post(args):
    from .pipeline import run
    res = run(
        args.nek,
        args.mesh,
        args.out,
        args.istart,
        args.iend,
        args.long,
        aoa_deg=args.aoa_deg,
    )
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


def _cmd_run(args):
    from pathlib import Path

    from .nek5000 import load_config, run_from_config

    conf = load_config(args.config)
    if _workflow(conf).startswith("channel"):
        from .kth_channel import run_channel_from_config

        paths = run_channel_from_config(
            args.config,
            build=args.build,
            rebuild=args.rebuild,
            prepare_only=args.prepare_only,
            skip_nek=args.skip_nek,
            skip_post=args.skip_post,
        )
        print(f"[channel-run] prepared {paths.root}")
        print(f"              config: {paths.config}")
        if paths.mat_file and not args.prepare_only and not args.skip_post:
            print(f"              post:   {paths.mat_file}")
        if not args.prepare_only and not args.skip_nek:
            print(f"              log:    {paths.logfile}")
        return

    paths = run_from_config(
        Path(args.config),
        build=args.build,
        rebuild=args.rebuild,
        prepare_only=args.prepare_only,
        skip_nek=args.skip_nek,
        skip_post=args.skip_post,
    )
    print(f"[run] prepared {paths.root}")
    print(f"      config: {paths.config}")
    if paths.stat_file and not args.prepare_only and not args.skip_post:
        print(f"      post:   {paths.stat_file}")
    if not args.prepare_only and not args.skip_nek:
        print(f"      log:    {paths.logfile}")


def _cmd_build(args):
    from .nek5000 import build_from_config, load_config

    conf = load_config(args.config)
    if _workflow(conf).startswith("channel"):
        from .kth_channel import build_channel_from_config

        paths = build_channel_from_config(args.config, rebuild=args.rebuild)
    else:
        paths = build_from_config(args.config, rebuild=args.rebuild)
    print(f"[build] source: {paths.source_dir}")
    print(f"        work:   {paths.work_dir}")
    print(f"        exe:    {paths.installed_executable}")
    print(f"        log:    {paths.logfile}")


def _cmd_clcd_backward(args):
    from .clcd_backward import run_clcd_backward_from_config

    paths = run_clcd_backward_from_config(
        args.config,
        build=args.build,
        rebuild=args.rebuild,
        prepare_only=args.prepare_only,
        skip_nek=args.skip_nek,
        force=args.force,
        discard_int_fld=args.discard_int_fld,
    )
    print(f"[clcd-backward] prepared {paths.root}")
    print(f"                config:   {paths.config}")
    print(f"                manifest: {paths.manifest}")
    if not args.prepare_only:
        print(f"                blocks:   {paths.block_csv}")
        print(f"                backward: {paths.csv}")


def _cmd_channel_post(args):
    from pathlib import Path

    from .channel_pipeline import run_channel_post_from_config
    from .nek5000 import load_config

    config_path = Path(args.config).resolve()
    conf = load_config(config_path)
    result = run_channel_post_from_config(
        conf,
        base_dir=config_path.parent,
        int_fld_override=args.int_fld,
        out_override=args.out,
        figdir_override=args.figdir,
        save_figures_override=False if args.no_figures else None,
    )
    print(f"[channel-post] int_fld: {result.int_fld}")
    if result.mat_file:
        print(f"               mat:     {result.mat_file}")
    if result.figdir and not args.no_figures:
        print(f"               figs:    {result.figdir}")
    print(
        f"               Retau={result.retau:.6g} "
        f"Reth={result.re_theta:.6g} H12={result.h12:.6g}"
    )


def _cmd_validate_channel(args):
    from .channel_pipeline import compare_channel_mat_files

    diffs = compare_channel_mat_files(args.mine, args.ref)
    print("Max |mine - ref| over common numeric channel data fields:")
    for name in sorted(diffs):
        print(f"   {name:28s} {diffs[name]:.3e}")


def _workflow(conf):
    return str(conf.get("workflow", "naca_wing")).lower().replace("-", "_")


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

    sb = sub.add_parser("build", help="extract solver archive and compile a case executable")
    sb.add_argument("config", help="YAML run configuration")
    sb.add_argument("--rebuild", action="store_true",
                    help="compile even if the configured output executable exists")
    sb.set_defaults(func=_cmd_build)

    sr = sub.add_parser("run", help="prepare a Nek run folder and run configured stages")
    sr.add_argument("config", help="YAML run configuration")
    sr.add_argument("--build", action="store_true", default=None,
                    help="build the configured Nek executable before staging the run")
    sr.add_argument("--rebuild", action="store_true",
                    help="force compilation when used with --build")
    sr.add_argument("--prepare-only", action="store_true",
                    help="stage the run folder, then stop")
    sr.add_argument("--skip-nek", action="store_true",
                    help="skip the Nek5000 interpolation stage")
    sr.add_argument("--skip-post", action="store_true",
                    help="skip Python int_fld post-processing")
    sr.set_defaults(func=_cmd_run)

    sp = sub.add_parser("post", help="process int_fld -> stat .mat")
    sp.add_argument("--nek", default=_default("nek_dir"),
                    help="dir containing ZSTAT/int_fld")
    sp.add_argument("--mesh", default=_default("mesh"), help="int_mesh.mat")
    sp.add_argument("--out", default="stat_naca4412_200k.mat")
    sp.add_argument("--istart", type=int, default=25)
    sp.add_argument("--iend", type=int, default=80)
    sp.add_argument("--long", type=int, default=1)
    sp.add_argument("--aoa-deg", type=float, default=5.0)
    sp.set_defaults(func=_cmd_post)

    sc = sub.add_parser(
        "clcd-backward",
        help="run blockwise Cl/Cd windows and write Python-only backward convergence CSVs",
    )
    sc.add_argument("config", help="YAML run configuration with clcd_backward section")
    sc.add_argument("--build", action="store_true", default=None,
                    help="build the configured Nek executable before staging")
    sc.add_argument("--rebuild", action="store_true",
                    help="force compilation when used with --build")
    sc.add_argument("--prepare-only", action="store_true",
                    help="stage the backward run folder and write the planned manifest")
    sc.add_argument("--skip-nek", action="store_true",
                    help="reuse existing per-block int_fld files")
    sc.add_argument("--force", action="store_true", default=None,
                    help="regenerate per-block int_fld files even if valid outputs exist")
    sc.add_argument("--discard-int-fld", action="store_true", default=None,
                    help="remove per-block int_fld files after writing the Cl/Cd CSVs")
    sc.set_defaults(func=_cmd_clcd_backward)

    sch = sub.add_parser("channel-post", help="process KTH channel int_fld -> MATLAB data .mat")
    sch.add_argument("config", help="YAML channel configuration")
    sch.add_argument("--int-fld", default=None,
                     help="override channel_post.int_fld")
    sch.add_argument("--out", default=None,
                     help="override channel_post.out")
    sch.add_argument("--figdir", default=None,
                     help="override channel_post.figdir")
    sch.add_argument("--no-figures", action="store_true",
                     help="write the .mat output without regenerating figures")
    sch.set_defaults(func=_cmd_channel_post)

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

    svc = sub.add_parser("validate-channel", help="diff two channel MATLAB data .mat files")
    svc.add_argument("--mine", required=True)
    svc.add_argument("--ref", required=True)
    svc.set_defaults(func=_cmd_validate_channel)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
