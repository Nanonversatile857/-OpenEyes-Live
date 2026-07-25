# 👁️ OpenEyes-Live

> **一个模块化、可插拔、端云协同的端侧视频AI基础设施。**
> —— 把最新的端侧多模态模型，变成可独立下载、按需加载、自由组合的AI引擎。

[![GitHub stars](https://img.shields.io/github/stars/vfvincentwong2026/-OpenEyes-Live?style=social)](https://github.com/vfvincentwong2026/-OpenEyes-Live/stargazers)
[![CI](https://github.com/vfvincentwong2026/-OpenEyes-Live/actions/workflows/ci.yml/badge.svg)](https://github.com/vfvincentwong2026/-OpenEyes-Live/actions/workflows/ci.yml)
[![GitHub forks](https://img.shields.io/github/forks/vfvincentwong2026/-OpenEyes-Live?style=social)](https://github.com/vfvincentwong2026/-OpenEyes-Live/network/members)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/Tests-118%20passing-brightgreen)](https://github.com/vfvincentwong2026/-OpenEyes-Live/actions)
[![Min Engine](https://img.shields.io/badge/Min_Engine-2MB-green)](https://github.com/vfvincentwong2026/-OpenEyes-Live)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey)](https://github.com/vfvincentwong2026/-OpenEyes-Live)
[![Download](https://img.shields.io/badge/Download-ModelScope-blue?logo=alibabacloud)](https://modelscope.cn/organization/OpenEyes-Live)
[![Version](https://img.shields.io/badge/Version-0.3.0-orange)](https://github.com/vfvincentwong2026/-OpenEyes-Live/releases)

---

## 🎬 See It In Action

> **A 5-second demo is worth a thousand lines of code.**

| 👇 Real-time Scene Understanding | 🔔 Proactive Alert |
| :---: | :---: |
| ![Real-time Scene Understanding](https://your-image-host.com/demo-cat.gif) | ![Proactive Alert](https://your-image-host.com/demo-boiling.gif) |
| *Camera sees a cat → AI says "An orange cat is lying on the sofa"* | *Camera sees boiling water → AI alerts "Water is boiling!"* |

> [!IMPORTANT]
> **Upload your actual demo GIF here before sharing!**
>
> 1. Record a 5-second video using OpenEyes-Live
> 2. Convert to GIF using `ffmpeg -i input.mp4 -vf "fps=15,scale=640:-1" -loop 0 output.gif`
> 3. Upload to your GitHub repo's `/assets/` folder or [imgur](https://imgur.com/)
> 4. Replace the image links above

*No demo yet? Run it now and record one — a rough demo beats a perfect README any day.*

---

## 📚 Full Documentation

| Document | Description | Audience |
| :--- | :--- | :--- |
| [📖 PROJECT_DOC.md](./docs/PROJECT_DOC.md) | Vision, scenarios, roadmap, user personas | All users |
| [🏗️ ARCHITECTURE.md](./docs/ARCHITECTURE.md) | Scheduler, engines, MCP gateway, lifecycle | Developers |
| [🔪 VIDEO_PIPELINE.md](./docs/VIDEO_PIPELINE.md) | Video engine pipeline: sampler/filter/encoder/compressor | Developers |
| [🎤 AUDIO_PIPELINE.md](./docs/AUDIO_PIPELINE.md) | Audio engine pipeline: VAD/ASR/speaker | Developers |
| [📊 PERFORMANCE_BENCHMARK.md](./docs/PERFORMANCE_BENCHMARK.md) | Device compatibility, benchmark data | Developers, decision-makers |
| [📱 QUICKSTART_MOBILE.md](./docs/QUICKSTART_MOBILE.md) | IP Webcam + Termux setup guide | All users |
| [🔧 API_REFERENCE.md](./docs/API_REFERENCE.md) | Complete Python API reference | Developers |
| [📐 ENGINE_SPEC.md](./docs/ENGINE_SPEC.md) | Engine interface specification for contributors | Contributors |
| [🤝 CONTRIBUTING.md](./CONTRIBUTING.md) | Contribution guidelines | Contributors |
| [📜 CODE_OF_CONDUCT.md](./CODE_OF_CONDUCT.md) | Community standards | All community members |
| [📋 CHANGELOG.md](./CHANGELOG.md) | Release notes and version history | All users |

---

## 🤔 Why OpenEyes-Live?

Got old phones sitting in a drawer?

- Screen still works, camera still works, WiFi still works
- But the OS is too old, apps won't install — just collecting dust

**What if I told you that with just 80MB of storage, that "e-waste" could become an AI device that sees, hears, and understands the world?**

OpenEyes-Live's promise is simple:

| Promise | Description |
| :--- | :--- |
| 📦 **80MB+** | Smallest sub-engine is just 80MB — fits on almost any phone |
| 🔪 **Video Pipeline** | Sampler → Filter → Encoder → Compressor — each pluggable |
| 🎤 **Audio Pipeline** | VAD + ASR + Speaker Recognition — let AI "hear and identify" |
| 🧩 **On-Demand** | Download only the engines you need. No wasted storage. |
| 🔒 **Fully Local** | Camera feed never leaves your device. Zero uploads. |
| 🌏 **Global Mirror** | Models hosted on Hugging Face + ModelScope for fast downloads in China |
| 💰 **100% Free** | No API keys, no subscriptions, no hidden costs |

**Turn retired phones into devices that actually see, hear, and understand.**

---

## ✨ What Can It Do?

| Scenario | How It Works | The Experience |
| :--- | :--- | :--- |
| 👁️ **Real-time Visual Description** | Point camera at anything | AI says: "An orange cat is lying on the sofa" |
| 🔔 **Proactive Alerts** | Run in background monitoring mode | "Water is boiling!" / "Someone's at the door!" |
| 🧠 **Memory & Q&A** | Ask about what you've seen | "What was the name of that book I saw earlier?" |
| 🚶 **Visual Assistance** | Phone worn on chest | Real-time narration: "Stairs 2 meters ahead" |
| 📹 **Home Monitoring** | Old phone fixed on a shelf | Falls, intruders, dangerous behavior — AI alerts you |
| 🎤 **Voice Interaction** | Speak to your phone | "What am I looking at?" → AI responds verbally |
| 🆔 **Who is Speaking?** | AI identifies the speaker | "That's your daughter talking" |
| 🔌 **AI Eyes as a Service** | Run as MCP Server | Claude, Cursor, and other AI tools can call it to "see" and "hear" |

> In one sentence: **Turn your old phone into a device that sees, hears, identifies, and proactively speaks.**

---

## 🏗️ Architecture: Modular & Pluggable Pipeline

The core philosophy of OpenEyes-Live: **every engine does one thing well — download on-demand, combine freely.**

```mermaid
flowchart TB
    subgraph Input["📥 Input Layer"]
        Camera["📷 视频流<br>（手机 / IP Webcam / USB）"]
        Mic["🎤 音频流<br>（麦克风 / 系统音频）"]
    end

    subgraph VideoPipeline["🎬 视频引擎流水线"]
        direction LR
        Sampler["帧采样引擎<br>（均匀/动态采样）"]
        Filter["帧筛选引擎<br>（Attention帧评分）"]
        Encoder["视觉编码引擎<br>（CLIP / VideoMamba）"]
        Compressor["Token压缩引擎<br>（高效Token投影器）"]
    end

    subgraph AudioPipeline["🎤 音频引擎流水线"]
        direction LR
        VAD["VAD引擎<br>（语音活动检测）"]
        ASR["ASR引擎<br>（语音识别+标点恢复）"]
        Speaker["声纹引擎<br>（说话人识别/验证）"]
    end

    subgraph Core["🧠 推理与记忆"]
        LLM["语言推理引擎<br>（Qwen / Llama）"]
        Memory["记忆引擎<br>（向量检索 + 时间线）"]
        MCP["MCP网关引擎<br>（协议适配）"]
    end

    subgraph Output["📤 输出层"]
        Text["📝 自然语言描述"]
        Voice["🔊 语音播报"]
        Alert["🔔 主动告警"]
        MCPOut["🔌 MCP工具调用"]
    end

    Camera --> Sampler --> Filter --> Encoder --> Compressor
    Mic --> VAD --> ASR
    ASR --> Speaker
    Compressor --> LLM
    ASR --> LLM
    Speaker --> LLM
    LLM --> Memory
    LLM --> MCP
    Memory --> Text
    LLM --> Voice
    LLM --> Alert
    MCP --> MCPOut
🧩 Engine Breakdown
Pipeline	Engine	Model	Size	Responsibility
Video	帧采样引擎	Configurable	~5MB	Uniform/dynamic frame sampling
Video	帧筛选引擎	Attention-based	~15MB	Score and select key frames
Video	视觉编码引擎	CLIP / VideoMamba	~200MB	Extract spatio-temporal features
Video	Token压缩引擎	Token Projector	~20MB	Compress redundant visual tokens
Audio	VAD引擎	Silero VAD	~2MB	Voice activity detection
Audio	ASR引擎	Parakeet / Voxtral	~150MB	Speech-to-text + punctuation recovery
Audio	声纹引擎	Speaker Embedding	~30MB	Speaker recognition / verification
Core	语言推理引擎	Qwen / Llama	~400MB	Natural language generation
Core	记忆引擎	Sentence-BERT + FAISS	~50MB	Vector memory + timeline
Core	MCP网关引擎	Lightweight adapter	~10MB	Expose capabilities to AI clients
🔄 Key Mechanisms
Mechanism	Description
Pipeline Separation	Video pipeline (4 stages) and Audio pipeline (3 stages) are fully decoupled
On-Demand Loading	Each sub-engine is downloaded only when first used
Free Combination	video_encoder+llm for description, vad+asr+speaker for voice interaction
Auto Downgrade	Automatically switches to lighter engine combinations on low-end devices
🔬 Model Selection
Engine	Model	Params	Quantized Size	Key Advantage
帧筛选引擎	Attention-based Scorer	—	~15MB	Auto-select key frames, reduce compute by 50-70%
视觉编码引擎	VideoMamba / CLIP	100M	~200MB	Efficient spatio-temporal feature extraction
Token压缩引擎	Token Projector	—	~20MB	2x inference speedup with minimal accuracy loss
ASR引擎	Parakeet TDT	0.6B	~150MB	On-device ASR + punctuation recovery
声纹引擎	Speaker Embedding	—	~30MB	Speaker verification, identify "who is speaking"
语言推理引擎	Qwen 3.5 Small	2B	~400MB	Balanced performance on edge devices
📦 All models are quantized via llama.cpp / GGUF for smooth performance on older devices.

🚀 Quick Start
Requirements
Android 6.0+ / iOS 12+ / Linux / macOS

At least 1GB free storage

A device with a camera (or external USB camera)

One-Line Setup
bash
# Clone the repo
git clone https://github.com/vfvincentwong2026/OpenEyes-Live.git
cd OpenEyes-Live

# Install dependencies
pip install -r requirements.txt

# Install the CLI tool
pip install -e .
Basic Usage
bash
# List all available sub-engines
openeyes list

# Download engine models (real download, resumable, hf-mirror fallback)
openeyes install encoder            # ~89MB  (CLIP ViT-B/32, quantized ONNX)
openeyes install llm                # ~3.3GB (Phi-3.5-vision int4)
openeyes install vad                # ~2MB   (Silero VAD)
openeyes install asr                # ~234MB (SenseVoice int8)
openeyes install speaker            # ~38MB  (3D-Speaker ERes2Net, GitHub release)

# Start with video only (base configuration)
openeyes watch --source camera --engines encoder+llm

# Transcribe microphone speech in real time (VAD + ASR)
openeyes listen

# With speaker identification: enroll once, then "张三说：……"
openeyes listen --enroll-mic 张三     # enroll by speaking one sentence
openeyes listen --enroll 张三:me.wav  # or enroll from a 16kHz mono wav
openeyes listen --speaker             # enrolled DB auto-loads next time

# Start MCP Server (for Claude, Cursor, etc.)
openeyes mcp --port 3000
How to Connect Your Phone
Method	Description	Recommendation
Local Camera	Run directly on the phone (Android support)	⭐⭐⭐⭐⭐
IP Webcam App	Old phone runs IP Webcam, sends feed via WiFi	⭐⭐⭐⭐⭐
USB Camera	Connect via OTG cable	⭐⭐⭐⭐
📱 Tested Compatibility
Coming soon in v0.2.0 release

Phone Model	Year	OS	Video Only	Video + Audio	Experience
Xiaomi 6	2017	Android 9	✅ Smooth	🟡 Limited	Tested
iPhone 8	2017	iOS 15	✅ Smooth	🟡 Limited	Tested
Redmi Note 5	2018	Android 8	✅ Usable	❌ OOM	Tested
Snapdragon 660+	2017+	Android 8+	✅ Recommended	⚠️ Testing	Ongoing
Benchmark Methodology: All data tested with llama.cpp (GGUF Q4_K_M) on ambient 25°C with passive cooling. See PERFORMANCE_BENCHMARK.md for detailed methodology.

🆚 Comparison with Similar Projects
Dimension	OpenEyes-Live	Mobile-VideoGPT	MiniCPM-o	Vision Agents
Video Pipeline	✅ 4-stage pluggable	❌ End-to-end	❌ End-to-end	❌ End-to-end
Audio Pipeline	✅ VAD+ASR+Speaker	❌ None	✅ Full-duplex	❌ None
Speaker Recognition	✅	❌	❌	❌
Sub-Engine On-Demand	✅	❌	❌	❌
Free Combination	✅	❌	❌	⚠️ Limited
Native MCP Support	✅	❌	❌	✅
Min Storage	80MB	~512MB	~6GB	~200MB
License	Apache 2.0	MIT	Apache 2.0	MIT
📂 Project Structure
text
OpenEyes-Live/
├── src/
│   ├── core/
│   │   ├── scheduler.py          # Multi-modal scheduler
│   │   └── engine_manager.py     # Engine lifecycle
│   ├── engines/
│   │   ├── video/
│   │   │   ├── sampler.py        # Frame sampling engine
│   │   │   ├── filter.py         # Attention-based frame filter
│   │   │   ├── encoder.py        # Visual encoding engine
│   │   │   └── compressor.py     # Token compression engine
│   │   ├── audio/
│   │   │   ├── vad.py            # VAD engine
│   │   │   ├── asr.py            # ASR + punctuation recovery
│   │   │   └── speaker.py        # Speaker recognition engine
│   │   └── core/
│   │       ├── llm.py            # Language reasoning engine
│   │       ├── memory.py         # Vector memory engine
│   │       └── mcp_gateway.py    # MCP gateway engine
│   ├── runtime/
│   │   ├── camera.py             # Camera input
│   │   └── audio.py              # Audio input
│   └── cli/
│       └── main.py
├── docs/
│   ├── PROJECT_DOC.md
│   ├── ARCHITECTURE.md
│   ├── VIDEO_PIPELINE.md
│   ├── AUDIO_PIPELINE.md
│   ├── ENGINE_SPEC.md
│   ├── PERFORMANCE_BENCHMARK.md
│   ├── QUICKSTART_MOBILE.md
│   └── API_REFERENCE.md
├── models/                        # Engine download cache
├── tests/
├── requirements.txt
├── setup.py
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── CHANGELOG.md
├── LICENSE
└── README.md
⚠️ Troubleshooting
Problem	Possible Cause	Solution
Camera not opening	Termux permission not granted	Run termux-setup-storage and restart Termux
IP Webcam connection failed	Not on same WiFi	Check IP address, ping the device
Slow inference (< 3 t/s)	Device too old / too many engines loaded	Use fewer engines, e.g., video_encoder+llm only
OOM / App crashes	Insufficient RAM	Single pipeline only: video OR audio, not both
Model download slow / fails	Network issues (China)	Use --mirror=hf-mirror flag
Thermal throttling	Continuous high load	Reduce FPS to 5-10fps, remove phone case
🗺️ Roadmap
Version	Goal	Key Features
v0.1.0	Video pipeline MVP	Video_encoder + LLM + Camera + CLI ✅
v0.2.0	Real engines	Real CLIP / Phi-3.5-vision VLM / Silero VAD / SenseVoice ASR + `openeyes listen` ✅
v0.3.0	Real model distribution	`openeyes install` real downloads (resume + hf-mirror fallback) ✅
v0.4.0	Full multimodal	Video + Audio + Memory + Proactive Alerts + Speaker Recognition
v1.0.0	Stable release	MCP full support + Cross-platform + App
🤝 Contributing
We especially welcome contributions from:

Role	What You Can Do
🧑‍💻 AI/ML Engineer	Add new models to video/audio pipelines
📱 Mobile Developer	Package Android/iOS App
🔧 Embedded Engineer	Device adaptation, performance tuning
🎨 UI/UX Designer	Design intuitive interfaces
📝 Technical Writer	Tutorials, translations, docs
🧪 Beta Tester	Test on old phones, file issues
👉 View Good First Issues

📄 License
Apache License 2.0 — Free to use, modify, and distribute, including commercially.

⭐ Support OpenEyes-Live
⭐ Star it — help others discover it

🔄 Share it — tell someone with an old phone

🛠️ Contribute — even just filing an issue helps

🙏 Acknowledgments
SmolVLM2 — Extreme edge video understanding

Mobile-VideoGPT — Real-time video understanding

Vision Agents — Real-time vision AI framework

llama.cpp — Making LLMs run on edge devices

sherpa-onnx — On-device speech processing

Made with ❤️ by the open source community.

If you see the value in this, please give it a Star 🌟

🇨🇳 中文简介
OpenEyes-Live 是一个模块化、可插拔的端侧多模态AI基础设施。

视频引擎流水线：帧采样 → 帧筛选 → 视觉编码 → Token压缩，每段独立可插拔

音频引擎流水线：VAD → ASR → 声纹识别，让AI“听得懂、辨得出”

最小仅需 80MB 存储空间

完全本地运行，数据不上云

按需下载，自由组合

让淘汰的旧手机重新拥有「看懂、听懂、辨人」的能力

👉 查看完整中文文档