import { joinSession } from "@github/copilot-sdk/extension";
import { execFile } from "child_process";
import { promisify } from "util";
import path from "path";
import { fileURLToPath } from "url";

const execFileAsync = promisify(execFile);
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PROJECT_ROOT = path.resolve(__dirname, "..", "..", "..");

async function runPython(args) {
    const python = process.platform === "win32" ? "python" : "python3";
    const { stdout, stderr } = await execFileAsync(
        python,
        ["-m", "zhrreport.cli", ...args],
        {
            cwd: PROJECT_ROOT,
            timeout: 1_800_000,
            maxBuffer: 8 * 1024 * 1024,
        },
    );
    return [stdout.trim(), stderr.trim()].filter(Boolean).join("\n");
}

function add(args, option, value) {
    if (value !== undefined && value !== null && value !== "") {
        args.push(option, String(value));
    }
}

const session = await joinSession({
    tools: [
        {
            name: "zhrreport_generate",
            description:
                "Generate separate Markdown and DOCX evidence reports from MP3, WAV, or JSON. " +
                "For audio without a known range, call zhrreport_timeline first and confirm a range.",
            parameters: {
                type: "object",
                properties: {
                    source: { type: "string", description: "Path to MP3, WAV, or transcript JSON." },
                    output_dir: { type: "string", description: "Report output directory (default: ./output)." },
                    range: { type: "string", description: "Confirmed START-END range, e.g. 00:05:26-00:07:26." },
                    confirm_full: { type: "boolean", description: "Process the entire source without prompting." },
                    backend: {
                        type: "string",
                        enum: ["json", "fixture", "faster-whisper"],
                        description: "Use faster-whisper for MP3/WAV; json for transcript JSON.",
                    },
                    model: { type: "string", description: "Local Faster-Whisper model (default: base)." },
                    fallback_model: { type: "string", description: "Optional larger model for bad chunks." },
                    language: { type: "string", description: "Optional ASR language code such as zh or en." },
                    speaker_name: { type: "string", description: "Verified name for the selected range." },
                    chunk_duration: { type: "number", description: "Worker chunk seconds; use 30-60 (default: 45)." },
                    timeout: { type: "number", description: "Hard timeout seconds per worker (default: 120)." },
                    diarization_backend: {
                        type: "string",
                        enum: ["numpy_acoustic", "speechbrain", "pyannote"],
                        description: "No-range audio diarization backend.",
                    },
                    num_speakers: { type: "integer", description: "Expected speakers for diarization (default: 2)." },
                    skip_hallucination: { type: "boolean", description: "Skip report hallucination flags." },
                },
                required: ["source"],
            },
            handler: async (input) => {
                const args = ["report", input.source];
                add(args, "--output-dir", input.output_dir);
                add(args, "--range", input.range);
                add(args, "--backend", input.backend);
                add(args, "--model", input.model);
                add(args, "--fallback-model", input.fallback_model);
                add(args, "--language", input.language);
                add(args, "--speaker-name", input.speaker_name);
                add(args, "--chunk-duration", input.chunk_duration);
                add(args, "--timeout", input.timeout);
                add(args, "--diarization-backend", input.diarization_backend);
                add(args, "--num-speakers", input.num_speakers);
                if (input.confirm_full) args.push("--confirm-full");
                if (input.skip_hallucination) args.push("--skip-hallucination");
                return (await runPython(args)) || "Report generation complete.";
            },
        },
        {
            name: "zhrreport_timeline",
            description:
                "Show anonymous candidate speaker ranges for MP3/WAV using executable local diarization, " +
                "or show the labeled timeline from transcript JSON.",
            parameters: {
                type: "object",
                properties: {
                    source: { type: "string", description: "Path to MP3, WAV, or transcript JSON." },
                    range: { type: "string", description: "Optional audio sub-range to diarize." },
                    diarization_backend: {
                        type: "string",
                        enum: ["numpy_acoustic", "speechbrain", "pyannote"],
                        description: "Audio diarization backend (default: numpy_acoustic).",
                    },
                    num_speakers: { type: "integer", description: "Expected number of speakers (default: 2)." },
                    output_dir: { type: "string", description: "Optional directory for diarization_timeline.txt." },
                },
                required: ["source"],
            },
            handler: async (input) => {
                if (path.extname(input.source).toLowerCase() === ".json") {
                    return await runPython(["timeline", input.source]);
                }
                const args = ["diarize", input.source];
                add(args, "--range", input.range);
                add(args, "--diarization-backend", input.diarization_backend);
                add(args, "--num-speakers", input.num_speakers);
                add(args, "--output-dir", input.output_dir);
                return await runPython(args);
            },
        },
        {
            name: "zhrreport_check_hallucination",
            description:
                "Run rule-based hallucination detection on transcript JSON or a bounded MP3/WAV range.",
            parameters: {
                type: "object",
                properties: {
                    source: { type: "string", description: "Path to transcript JSON, MP3, or WAV." },
                    range: { type: "string", description: "Required START-END range for audio." },
                    model: { type: "string", description: "Faster-Whisper model (default: base)." },
                    fallback_model: { type: "string", description: "Optional model for suspect chunks." },
                    language: { type: "string", description: "Optional language code such as zh or en." },
                    speaker_name: { type: "string", description: "Speaker label for the audio range." },
                    chunk_duration: { type: "number", description: "Worker chunk seconds (default: 45)." },
                    timeout: { type: "number", description: "Hard timeout per worker (default: 120)." },
                },
                required: ["source"],
            },
            handler: async (input) => {
                const args = ["check-hallucination", input.source];
                add(args, "--range", input.range);
                add(args, "--model", input.model);
                add(args, "--fallback-model", input.fallback_model);
                add(args, "--language", input.language);
                add(args, "--speaker-name", input.speaker_name);
                add(args, "--chunk-duration", input.chunk_duration);
                add(args, "--timeout", input.timeout);
                return await runPython(args);
            },
        },
        {
            name: "zhrreport_backends",
            description: "List ASR and diarization backends with real dependency availability/install hints.",
            parameters: { type: "object", properties: {} },
            handler: async () => {
                const asr = await runPython(["backends"]);
                const diarizers = await runPython(["diarizers"]);
                return `${asr}\n\n${diarizers}`;
            },
        },
    ],
    hooks: {
        onSessionStart: async () => {
            await session.log("zhrreport extension loaded with MP3/WAV, local ASR, and diarization support");
        },
    },
});
