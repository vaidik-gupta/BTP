# pip install pylsl pandas

from pylsl import StreamInlet, resolve_stream
import pandas as pd
import time
from datetime import datetime

# =========================
# SETTINGS
# =========================
SAVE_SECONDS = 600        # how long to record (10 min)
OUTPUT_FILE = f"xon_data_2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

# =========================
# FIND LSL STREAM
# =========================
print("Searching for LSL stream from Xon headset...")

streams = resolve_stream()

if len(streams) == 0:
    raise Exception("No LSL streams found. Make sure Xon app is running and streaming.")

print("\nAvailable Streams:")
for i, s in enumerate(streams):
    print(f"{i}: Name={s.name()} | Type={s.type()} | Channels={s.channel_count()}")

# Use first stream
stream = streams[0]

print(f"\nConnecting to stream: {stream.name()}")

inlet = StreamInlet(stream)

# =========================
# STREAM INFO
# =========================
channel_count = stream.channel_count()
sampling_rate = stream.nominal_srate()

print(f"Channels      : {channel_count}")
print(f"Sampling Rate : {sampling_rate} Hz")

# =========================
# DATA COLLECTION
# =========================
data = []

start_time = time.time()

print("\nRecording started...")
print(f"Saving to: {OUTPUT_FILE}\n")

sample_index = 0

try:
    while time.time() - start_time < SAVE_SECONDS:

        # sample = EEG values
        # timestamp = exact LSL timestamp
        sample, timestamp = inlet.pull_sample()

        row = {
            "sample_index": sample_index,
            "lsl_timestamp": timestamp,
            "pc_time_unix": time.time(),
            "pc_time_readable": datetime.now().isoformat()
        }

        # add all channels
        for ch in range(channel_count):
            row[f"ch_{ch+1}"] = sample[ch]

        data.append(row)

        sample_index += 1

        # print every 250 samples
        if sample_index % 250 == 0:
            print(f"Collected {sample_index} samples")

except KeyboardInterrupt:
    print("\nStopped manually")

# =========================
# SAVE CSV
# =========================
df = pd.DataFrame(data)

df.to_csv(OUTPUT_FILE, index=False)

print("\nDone.")
print(f"Total samples collected: {len(df)}")
print(f"CSV saved as: {OUTPUT_FILE}")