import logging
import time

# pylint: disable=import-error
from ..functional import scalar_values
from .cif_hr import CifHr
from ..network import headmeta
from .. import visualizer

LOG = logging.getLogger(__name__)


class CifSeeds:
    threshold = None
    score_scale = 1.0
    debug_visualizer = visualizer.Seeds()

    def __init__(self, cifhr: CifHr):
        self.cifhr = cifhr
        self.seeds = []

    def fill(self, cifs, metas):
        for cif, meta in zip(cifs, metas):
            self.fill_single(cif, meta)
        return self

    def fill_single(self, cif, meta: headmeta.Intensity):
        start = time.perf_counter()

        sv = 0.0

        for field_i, p in enumerate(cif):
            if meta.decoder_seed_mask is not None and not meta.decoder_seed_mask[field_i]:
                continue
            p = p[:, p[0] > self.threshold]
            if meta.decoder_min_scale:
                p = p[:, p[4] > meta.decoder_min_scale / meta.stride]
            c, x, y, _, s = p

            start_sv = time.perf_counter()
            v = scalar_values(self.cifhr[field_i], x * meta.stride, y * meta.stride, default=0.0)
            v = 0.9 * v + 0.1 * c
            sv += time.perf_counter() - start_sv

            if self.score_scale != 1.0:
                v = v * self.score_scale
            m = v > self.threshold
            x, y, v, s = x[m] * meta.stride, y[m] * meta.stride, v[m], s[m] * meta.stride

            for vv, xx, yy, ss in zip(v, x, y, s):
                self.seeds.append((vv, field_i, xx, yy, ss))

        LOG.debug('seeds %d, %.3fs (C++ %.3fs)', len(self.seeds), time.perf_counter() - start, sv)
        return self

    def get(self):
        self.debug_visualizer.predicted(self.seeds)
        return sorted(self.seeds, reverse=True)


class CifDetSeeds(CifSeeds):
    def fill_single(self, cif, meta: headmeta.Detection):
        start = time.perf_counter()

        for field_i, p in enumerate(cif):
            p = p[:, p[0] > self.threshold]
            if meta.decoder_min_scale:
                p = p[:, p[4] > meta.decoder_min_scale / meta.stride]
                p = p[:, p[5] > meta.decoder_min_scale / meta.stride]
            c, x, y, _, w, h, _ = p
            v = scalar_values(self.cifhr[field_i], x * meta.stride, y * meta.stride, default=0.0)
            v = 0.9 * v + 0.1 * c
            if self.score_scale != 1.0:
                v = v * self.score_scale
            m = v > self.threshold

            x = x[m] * meta.stride
            y = y[m] * meta.stride
            v = v[m]
            w = w[m] * meta.stride
            h = h[m] * meta.stride

            for vv, xx, yy, ww, hh in zip(v, x, y, w, h):
                self.seeds.append((vv, field_i, xx, yy, ww, hh))

        LOG.debug('seeds %d, %.3fs', len(self.seeds), time.perf_counter() - start)
        return self
