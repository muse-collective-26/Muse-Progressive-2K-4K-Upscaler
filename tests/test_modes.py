"""Run with ComfyUI's Python and COMFYUI_PATH set to the ComfyUI directory.

Mock only heavy model execution. Check disabled branches, trimming, presets and
preflight limits; a real both-output GPU test is recorded separately.
"""
import importlib.util
import os
from pathlib import Path
import sys
import unittest
import tempfile
from unittest.mock import patch

sys.path.insert(0, os.environ['COMFYUI_PATH'])
root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('muse_test_package', root/'__init__.py', submodule_search_locations=[str(root)])
package = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = package
spec.loader.exec_module(package)
u = sys.modules['muse_test_package.upscaler']


class ModesTest(unittest.TestCase):
    def run_mode(self, mode, preset='Detailed · 0.80 (default)'):
        calls = []
        def fake_call(name, **kwargs):
            calls.append((name, kwargs))
            return ({'samples': None},)
        fake_models = dict(model=None, clip=None, vae=None, audio_vae=None)
        output_frames = u.torch.zeros((22, 2, 2, 3))
        tile = unittest.mock.Mock()
        tile.run.return_value = (output_frames, '{}')
        with patch.object(u.MuseProgressiveUpscaler, '_models', return_value=fake_models), \
             patch.object(u, 'call', side_effect=fake_call), \
             patch.object(u, 'decode', return_value=output_frames), \
             patch.object(u, 'resize_pixels', side_effect=lambda x, *args: x), \
             patch.object(u, 'Light4K', return_value=tile):
            result = u.MuseProgressiveUpscaler().run(u.torch.zeros((7, 128, 128, 3)), mode,
                preset, 42, '', '', '', '', '', u.DEFAULT_PROMPT)
        return result, tile, calls

    def test_2k_skips_4k_and_trims_padding(self):
        result, tile, calls = self.run_mode('2K only')
        tile.run.assert_not_called()
        self.assertEqual(len(result[0]), 7)
        self.assertIsInstance(result[1], u.ExecutionBlocker)
        ref = next(k for n, k in calls if n == 'MiniMaxH3ReferenceToVideo')
        self.assertEqual(ref['length'], 22)

    def test_4k_only_blocks_2k_save_but_uses_intermediate(self):
        result, tile, _ = self.run_mode('4K only')
        tile.run.assert_called_once()
        self.assertIsInstance(result[0], u.ExecutionBlocker)
        self.assertEqual(len(result[1]), 7)
        self.assertEqual(tile.run.call_args[0][0]['refine_sigmas'][0], .8)

    def test_both_presets_reach_both_stages(self):
        for preset, sigmas in u.PRESETS.items():
            result, tile, calls = self.run_mode('Both', preset)
            self.assertEqual(len(result[0]), 7)
            self.assertEqual(len(result[1]), 7)
            sample = next(k for n, k in calls if n == 'SamplerCustomAdvanced')
            self.assertAlmostEqual(sample['sigmas'][0].item(), sigmas[0])
            self.assertEqual(tile.run.call_args[0][0]['refine_sigmas'], sigmas)

    def test_long_clip_is_processed_without_a_frame_cap(self):
        node = u.MuseProgressiveUpscaler()
        sections = []
        def section(images, *args, **kwargs):
            sections.append(len(images))
            return images[:, :1, :1, :], u.ExecutionBlocker(None), '{}'
        source = u.torch.arange(244, dtype=u.torch.float32)[:, None, None, None].expand(-1, 1, 1, 3)
        with patch.object(node, '_models', return_value={}), patch.object(node, '_run_chunk', side_effect=section), \
             patch.object(u.folder_paths, 'get_temp_directory', return_value=tempfile.gettempdir()):
            result = node.run(source, '2K only', next(iter(u.PRESETS)),42,'','','','','','')
        self.assertEqual(sections, [124, 124, 6])
        self.assertEqual(len(result[0]), 244)
        self.assertTrue(u.torch.allclose(result[0], source))
        self.assertIsInstance(result[1], u.ExecutionBlocker)

    def test_tested_resolution(self):
        self.assertEqual(u.output_sizes(1344, 768), ((2528, 1440), (3840, 2208)))


if __name__ == '__main__':
    unittest.main()
