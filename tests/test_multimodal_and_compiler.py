"""
Unit tests for M-2LRF Multi-Modal, Compiler, Agent, and Evaluation Subsystems.
"""

import unittest
import torch
import torch.nn as nn

from m2lrf.multimodal import (
    LinearProjector,
    MLPProjector,
    PerceiverResampler,
    PixelShuffleProjector,
    VisionTransformerEncoder,
    AudioTransformerEncoder,
    MultiModalProcessor,
)
from m2lrf.compiler import (
    GraphOptimizer,
    StaticMemoryPlanner,
    KernelCodeGenerator,
)
from m2lrf.agents import (
    ToolCallingEngine,
    ToolDefinition,
    StructuredOutputMasker,
    ReActAgent,
)
from m2lrf.evaluation import (
    GSM8KEvaluator,
    HumanEvalEvaluator,
    MultipleChoiceEvaluator,
)
from m2lrf.models.zoo.deepseek_v3 import DeepSeekV3Config, DeepSeekV3ForCausalLM
from m2lrf.models.zoo.phi4 import Phi4Config, Phi4ForCausalLM
from m2lrf.models.zoo.whisper import WhisperConfig, WhisperForConditionalGeneration
from m2lrf.models.zoo.flux import FluxConfig, FluxTransformer


class TestMultiModalSubsystem(unittest.TestCase):
    def test_01_projectors(self):
        x = torch.randn(2, 16, 64)
        lin = LinearProjector(64, 128)
        out_lin = lin(x)
        self.assertEqual(out_lin.shape, (2, 16, 128))

        mlp = MLPProjector(64, 128)
        out_mlp = mlp(x)
        self.assertEqual(out_mlp.shape, (2, 16, 128))

        perceiver = PerceiverResampler(visual_dim=64, llm_dim=128, num_latents=8, num_heads=4)
        out_perc = perceiver(x)
        self.assertEqual(out_perc.shape, (2, 8, 128))

        x_spatial = torch.randn(2, 16, 32)
        pixel_shuf = PixelShuffleProjector(in_features=32, out_features=64, downsample_factor=2)
        out_shuf = pixel_shuf(x_spatial, height=4, width=4)
        self.assertEqual(out_shuf.shape, (2, 4, 64))

    def test_02_vision_and_audio_encoders(self):
        vit = VisionTransformerEncoder(
            image_size=56,
            patch_size=14,
            embed_dim=64,
            num_layers=2,
            num_heads=4,
            intermediate_dim=128,
        )
        pixels = torch.randn(2, 3, 56, 56)
        out_vit = vit(pixels)
        self.assertEqual(out_vit.shape[0], 2)
        self.assertEqual(out_vit.shape[2], 64)

        audio = AudioTransformerEncoder(
            in_mels=40,
            embed_dim=64,
            num_layers=2,
            num_heads=4,
            intermediate_dim=128,
        )
        mels = torch.randn(2, 40, 32)
        out_audio = audio(mels)
        self.assertEqual(out_audio.shape[0], 2)
        self.assertEqual(out_audio.shape[2], 64)

    def test_03_multimodal_processor(self):
        proc = MultiModalProcessor(image_size=112)
        img_t = proc.process_image(None)
        self.assertEqual(img_t.shape, (3, 112, 112))
        aud_t = proc.process_audio(None)
        self.assertEqual(aud_t.shape[0], 80)
        formatted = proc.format_multimodal_prompt("What is this?", has_image=True)
        self.assertTrue(formatted.startswith("<image>"))


class TestCompilerSubsystem(unittest.TestCase):
    def test_04_graph_optimizer(self):
        class SimpleNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(32, 32)
                self.lm_head = nn.Linear(32, 64)

            def forward(self, x):
                return self.lm_head(self.q_proj(x))

        net = SimpleNet()
        replaced = GraphOptimizer.replace_linear_with_m2lrf(net, target_modules=["q_proj"], exclude_modules=["lm_head"])
        self.assertEqual(replaced, 1)
        self.assertIn("M2LRFUnifiedLinear", str(type(net.q_proj)))
        self.assertIn("Linear", str(type(net.lm_head)))

        counts = GraphOptimizer.count_parameters(net)
        self.assertIn("total_parameters", counts)
        self.assertIn("trainable_parameters", counts)

    def test_05_memory_planner(self):
        planner = StaticMemoryPlanner()
        planner.record_tensor("act1", size_bytes=1000, birth_step=0, death_step=2)
        planner.record_tensor("act2", size_bytes=1500, birth_step=3, death_step=5)
        peak = planner.compute_peak_memory()
        naive = planner.compute_naive_total()
        self.assertEqual(peak, 1500)
        self.assertEqual(naive, 2500)
        self.assertGreater(planner.memory_savings_ratio(), 0.0)

    def test_06_kernel_codegen(self):
        triton_src = KernelCodeGenerator.generate_triton_gemv_source()
        self.assertIn("@triton.jit", triton_src)
        cuda_src = KernelCodeGenerator.generate_cuda_gemv_source()
        self.assertIn("__global__", cuda_src)


class TestAgentAndEvaluationSubsystem(unittest.TestCase):
    def test_07_tool_calling(self):
        engine = ToolCallingEngine()

        def add(a: int, b: int) -> int:
            """Adds two integers."""
            return a + b

        engine.register(add)
        schemas = engine.get_tools_schema()
        self.assertEqual(len(schemas), 1)
        self.assertEqual(schemas[0]["function"]["name"], "add")

        sample_completion = 'Thought: Let me add.\nAction: <tool_call>{"name": "add", "arguments": {"a": 5, "b": 7}}</tool_call>'
        calls = engine.parse_tool_calls(sample_completion)
        self.assertEqual(len(calls), 1)
        res = engine.execute_call(calls[0])
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["output"], 12)

    def test_08_structured_output_masker(self):
        masker = StructuredOutputMasker(vocab_size=10)
        logits = torch.zeros(1, 10)
        valid_tokens = {2, 4, 6}
        masked = masker.mask_logits(logits, valid_tokens)
        self.assertEqual(masked[0, 2].item(), 0.0)
        self.assertEqual(masked[0, 0].item(), -float("inf"))

    def test_09_react_agent(self):
        engine = ToolCallingEngine()

        def multiply(x: int, y: int) -> int:
            """Multiplies two numbers."""
            return x * y

        engine.register(multiply)

        steps = [
            'Thought: I need to multiply 4 and 5.\nAction: <tool_call>{"name": "multiply", "arguments": {"x": 4, "y": 5}}</tool_call>',
            'Thought: Now I know the answer.\nFinal Answer: 20',
        ]
        step_iter = iter(steps)

        def mock_llm(prompt: str) -> str:
            return next(step_iter)

        agent = ReActAgent(llm_fn=mock_llm, tools=engine, max_iterations=3)
        result = agent.run("What is 4 times 5?")
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["final_answer"], "20")

    def test_10_evaluation_suites(self):
        # GSM8K
        self.assertEqual(GSM8KEvaluator.extract_answer("The answer is 42"), "42")
        self.assertEqual(GSM8KEvaluator.extract_answer("Final value: \\boxed{128}"), "128")
        self.assertTrue(GSM8KEvaluator.evaluate_sample("\\boxed{35}", "35"))
        self.assertFalse(GSM8KEvaluator.evaluate_sample("\\boxed{30}", "35"))

        # HumanEval
        sample_code = "def add(a, b):\n    return a + b"
        sample_test = "assert add(2, 3) == 5\nassert add(-1, 1) == 0"
        self.assertTrue(HumanEvalEvaluator.execute_and_verify(sample_code, sample_test))

        # MultipleChoice
        self.assertEqual(MultipleChoiceEvaluator.extract_choice_letter("The correct option is (B)."), "B")


class TestNewModelZooArchitectures(unittest.TestCase):
    def test_11_deepseek_v3(self):
        cfg = DeepSeekV3Config(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
            kv_lora_rank=16,
            q_lora_rank=32,
            qk_rope_head_dim=16,
            v_head_dim=16,
            qk_nope_head_dim=16,
            n_routed_experts=4,
            num_experts_per_tok=2,
        )
        model = DeepSeekV3ForCausalLM(cfg)
        x = torch.randint(0, 128, (2, 8))
        out = model(x)
        self.assertEqual(out.shape, (2, 8, 128))

    def test_12_phi4(self):
        cfg = Phi4Config(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
        )
        model = Phi4ForCausalLM(cfg)
        x = torch.randint(0, 128, (2, 8))
        out = model(x)
        self.assertEqual(out.shape, (2, 8, 128))

    def test_13_whisper(self):
        cfg = WhisperConfig(
            vocab_size=128,
            num_mel_bins=40,
            d_model=64,
            encoder_layers=1,
            decoder_layers=1,
            encoder_attention_heads=4,
            decoder_attention_heads=4,
            decoder_ffn_dim=128,
            encoder_ffn_dim=128,
        )
        model = WhisperForConditionalGeneration(cfg)
        mel = torch.randn(2, 40, 32)
        dec_in = torch.randint(0, 128, (2, 8))
        out = model(mel, dec_in)
        self.assertEqual(out.shape, (2, 8, 128))

    def test_14_flux(self):
        cfg = FluxConfig(
            hidden_size=64,
            num_heads=4,
            head_dim=16,
            num_dual_layers=1,
            num_single_layers=1,
        )
        model = FluxTransformer(cfg)
        img = torch.randn(2, 8, 64)
        txt = torch.randn(2, 4, 64)
        out = model(img, txt)
        self.assertEqual(out.shape, (2, 8, 64))


if __name__ == "__main__":
    unittest.main()
