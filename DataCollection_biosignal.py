import time
import csv
from pylsl import StreamInlet, resolve_streams

# =====================================
# CONFIG
# =====================================

STREAM_NAME_KEYWORD = ""   # keep empty to connect to first stream
TIMEOUT = 10
RETRIES = 5

OUTPUT_FILE = "biosignal_data_final_left.csv"

TOTAL_DURATION = 600       # 10 minutes
STATE_DURATION = 5         # 5 sec REST / 5 sec CURL


# =====================================
# FIND AVAILABLE STREAMS
# =====================================

def find_stream(timeout=10):

    print(f"\n🔍 Searching for LSL streams (timeout={timeout}s)...")

    streams = resolve_streams(wait_time=timeout)

    if not streams:
        print("❌ No streams found.")
        return None

    print(f"\n✅ Found {len(streams)} stream(s):\n")

    for i, s in enumerate(streams):

        print(f"[{i}]")
        print(f"Name           : {s.name()}")
        print(f"Type           : {s.type()}")
        print(f"Channels       : {s.channel_count()}")
        print(f"Sampling Rate  : {s.nominal_srate()}")
        print()

    return streams


# =====================================
# CONNECT TO STREAM
# =====================================

def connect_stream():

    for attempt in range(RETRIES):

        print(f"\n⚡ Attempt {attempt + 1}/{RETRIES}")

        streams = find_stream(TIMEOUT)

        if streams:

            # Try keyword matching first
            if STREAM_NAME_KEYWORD != "":

                for s in streams:

                    if STREAM_NAME_KEYWORD.lower() in s.name().lower():

                        print(f"\n✅ Connecting to: {s.name()}\n")

                        return StreamInlet(s)

            # If no keyword -> user chooses stream
            print("\nAvailable Streams:\n")

            for i, s in enumerate(streams):

                print(f"[{i}] {s.name()} ({s.type()})")

            idx = int(input("\nSelect stream index: "))

            inlet = StreamInlet(streams[idx])

            print("\n✅ Stream connected successfully.\n")

            return inlet

        print("\n⏳ Retrying in 2 sec...\n")

        time.sleep(2)

    raise RuntimeError("❌ Could not find any LSL stream")


# =====================================
# DETERMINE CURRENT LABEL
# =====================================

def get_current_state(elapsed_time):

    cycle = int(elapsed_time // STATE_DURATION)

    if cycle % 2 == 0:
        return "REST"
    else:
        return "CURL"


# =====================================
# RECORD DATA
# =====================================

def record_data(inlet):

    print("\n🎯 Starting EEG recording...\n")

    print("Protocol:")
    print("REST  -> 5 sec")
    print("CURL  -> 5 sec")
    print(f"Total Duration -> {TOTAL_DURATION} sec\n")

    experiment_start = time.time()

    current_state = None

    with open(OUTPUT_FILE, mode='w', newline='') as file:

        writer = csv.writer(file)

        # Pull first sample
        first_sample, ts = inlet.pull_sample(timeout=5)

        if first_sample is None:
            raise RuntimeError("❌ No data received from stream")

        num_channels = len(first_sample)

        # CSV HEADER
        header = [
            "lsl_timestamp",
            "local_timestamp",
            "elapsed_time_sec",
            "label"
        ] + [f"ch{i}" for i in range(num_channels)]

        writer.writerow(header)

        print(f"📡 Connected.")
        print(f"📊 Number of channels: {num_channels}")
        print(f"💾 Saving to: {OUTPUT_FILE}\n")

        try:

            while True:

                now = time.time()

                elapsed = now - experiment_start

                # Stop after total duration
                if elapsed >= TOTAL_DURATION:

                    print("\n✅ 10-minute recording completed.")

                    break

                # Determine current state
                state = get_current_state(elapsed)

                # Receive EEG sample
                sample, timestamp = inlet.pull_sample(timeout=2)

                if sample is None:
                    continue

                local_time = time.time()

                # Save sample
                writer.writerow([
                    timestamp,
                    local_time,
                    elapsed,
                    state,
                    *sample
                ])

                # PRINT ONLY WHEN STATE CHANGES
                if state != current_state:

                    current_state = state

                    print(
                        f"\n[{elapsed:7.2f} sec] "
                        f"=====================> {state}"
                    )

        except KeyboardInterrupt:

            print("\n🛑 Recording stopped manually.")

    print(f"\n📁 Data saved successfully -> {OUTPUT_FILE}")


# =====================================
# MAIN
# =====================================

if __name__ == "__main__":

    inlet = connect_stream()

    record_data(inlet)