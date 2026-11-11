"""layout.py must equal diffusers' own MiniMax-H3-Image helpers bit for bit (needs torch + diffusers)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
bd = pytest.importorskip("diffusers.modular_pipelines.minimax_h3_image.before_denoise")
from diffusers import MiniMaxH3Scheduler  # noqa: E402

from minimax_h3_mlx import layout as L  # noqa: E402

SIZES = [(3552, 4736), (4096, 4096), (1024, 1024), (768, 1360), (2688, 6272), (5472, 3072), (500, 333)]


@pytest.mark.parametrize("h,w", SIZES)
@pytest.mark.parametrize("area", [L.DEFAULT_MIN_AREA, 1024])
def test_canvas(h, w, area):
    assert L.canvas_size(h, w, area) == bd.get_image_canvas_size(h, w, area, 32)


@pytest.mark.parametrize("h,w", SIZES[:5])
def test_packed_layout(h, w):
    ch, cw = L.canvas_size(h, w)
    lh, lw = ch // 16, cw // 16
    tags = torch.tensor([1] * 37 + [0] * 5 + [1] * 9)
    refs = [(64, 96), (128, 80)]
    pos, tt, ii, ti, nref = bd.build_image_packed_sequence(tags, refs, lh, lw, (1, 2, 2), 0)
    lay = L.Layout(n_text=len(tags), text_tags=tags.numpy(), reference_grids=refs, latent_h=lh, latent_w=lw)
    assert (lay.position_ids() == pos.numpy()).all()
    assert (lay.tags() == tt.numpy()).all()
    assert lay.n_reference_rows == nref and lay.target_start == ii[nref].item()


def test_sampling_grid():
    sched = MiniMaxH3Scheduler(shift_provided_sigmas=False, euler_update_form="velocity")
    timesteps = torch.tensor([1.0 - s for s in L.SAMPLE_SIGMAS[:-1]], dtype=torch.float32)
    sched.set_timesteps(sigmas=torch.cat([1.0 - timesteps, torch.zeros(1)]))
    t, sigmas = L.sampling_grid()
    assert (sched.timesteps.numpy() == t).all() and (sched.sigmas.numpy() == sigmas).all()


def test_velocity_step_and_patchify():
    rng = np.random.default_rng(0)
    lat = rng.standard_normal((1, 24, 1, 12, 20)).astype(np.float32)
    rows = bd.patchify_video_latents(torch.from_numpy(lat), (1, 2, 2)).numpy()
    assert (rows == L.patchify(lat[0, :, 0])).all()
    assert (L.unpatchify(rows, 12, 20) == lat[:, :, 0]).all()
    sched = MiniMaxH3Scheduler(shift_provided_sigmas=False, euler_update_form="velocity")
    t, sigmas = L.sampling_grid()
    sched.set_timesteps(sigmas=torch.from_numpy(sigmas))
    v = rng.standard_normal(rows.shape).astype(np.float32)
    x = torch.from_numpy(rows)
    ours = rows.copy()
    for i, ti in enumerate(sched.timesteps):
        x = sched.step(torch.from_numpy(v), ti, x, return_dict=False)[0]
        ours = ours + (np.float32(sigmas[i]) - np.float32(sigmas[i + 1])) * v
        assert (x.numpy() == ours).all()
