import argparse
from projectaria_tools.utils.vrs_to_mp4_utils import convert_vrs_to_mp4

class VrsToMp4:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        if output_path == None:
            self.output_path = input_path.split(".")[0] + ".mp4"
        else:
            self.output_path = output_path
        print("Saving output to path: ", self.output_path)
        convert_vrs_to_mp4(self.input_path, self.output_path)
        print(f"Conversion complete. Output saved to {self.output_path}.")


def main():
    parser = argparse.ArgumentParser(description="Convert .vrs files to .mp4 format.")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to the input .vrs file.")
    parser.add_argument("--output", "-o", type=str, required=False, help="Path to save the output .mp4 file.") # by default the output path is the same as input path.
    args = parser.parse_args()
    converter = VrsToMp4(args.input, args.output)
if __name__ == "__main__":
    main()