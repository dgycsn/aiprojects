import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import whisper  # for Python-only audio loading

#%%
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

#taken from https://huggingface.co/Flurin17/whisper-large-v3-turbo-swiss-german
model_id = "Flurin17/whisper-large-v3-turbo-swiss-german"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, 
    dtype=torch_dtype, 
    low_cpu_mem_usage=True, 
    use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    dtype=torch_dtype,
    device=device,
)

#%%

file = "C:/Users/doga_/Downloads/swissgerman.mp4"

audio_array = whisper.load_audio(file)  # float32 array
audio_array = whisper.pad_or_trim(audio_array)  # optional, makes length suitable
sampling_rate = 16000  # Whisper default

#%%

result = pipe({
    "array": audio_array,
    "sampling_rate": sampling_rate,
    
    },
    generate_kwargs={
    "language": "de",      # force German
    "task": "transcribe"   # no translation
})

print(result["text"])
# output: Ich habe wirklich keinen Käskuchen gegessen