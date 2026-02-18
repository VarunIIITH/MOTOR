import sys
import av
import pandas as pd
import numpy as np
import cv2

camera_matrix = np.asarray([
        [
            761.2130214588122,
            0.0,
            531.6603272496533
        ],
        [
            0.0,
            760.5610942002685,
            547.0482801436125
        ],
        [
            0.0,
            0.0,
            1.0
        ]
    ])
dist_coeffs = np.asarray(  [
        [
            -0.1258081758854954,
            0.10090612036647595,
            0.00016502368429395915,
            0.0002943161078373772,
            0.014027441071045315,
            0.2021422493435316,
            0.009977388761135192,
            0.059168681863294686
        ]])
		
def undistort_video(original_video_path, undistorted_video_path):
	original_container = av.open(str(original_video_path))
	original_video_stream = original_container.streams.video[0]
	undistorted_container = av.open(str(undistorted_video_path), "w")

	# try:
	#     undistorted_video = undistorted_container.add_stream("h264_nvenc")
	# except Exception as e:
	#     print("nvenc not available", e)
	#     undistorted_video = undistorted_container.add_stream("h264")

	undistorted_video = undistorted_container.add_stream("h264")
	undistorted_video.options["bf"] = "0"
	undistorted_video.options["movflags"] = "faststart"
	# undistorted_video.gop_size = original_video_stream.gop_size
	undistorted_video.codec_context.height = original_video_stream.height
	undistorted_video.codec_context.width = original_video_stream.width
	undistorted_video.codec_context.time_base = original_video_stream.time_base
	undistorted_video.codec_context.bit_rate = original_video_stream.bit_rate
	print('Hello')

	if original_container.streams.audio:
		audio_stream = original_container.streams.audio[0]
		output_audio_stream = undistorted_container.add_stream("aac")
		output_audio_stream.codec_context.layout = audio_stream.layout.name
		output_audio_stream.codec_context.time_base = audio_stream.time_base
		output_audio_stream.codec_context.bit_rate = audio_stream.bit_rate
		output_audio_stream.codec_context.sample_rate = audio_stream.sample_rate

	with undistorted_container:
		index = 0
		for packet in original_container.demux():
			frames = packet.decode()

			if packet.stream.type == "audio":
				for frame in frames:
					packets = output_audio_stream.encode(frame)
					undistorted_container.mux(packets)
			elif packet.stream.type == "video": 
				for frame in frames:
					img = frame.to_ndarray(format="bgr24")
					undistorted_img = cv2.undistort(img, camera_matrix, dist_coeffs)
					new_frame = frame.from_ndarray(undistorted_img, format="bgr24")
					new_frame.pts = frame.pts
					new_frame.time_base = original_video_stream.time_base
					packets = undistorted_video.encode(new_frame)
					undistorted_container.mux(packets)
		
		# encode and mux frames that have been queued internally by the encoders
		undistorted_container.mux(output_audio_stream.encode())
		undistorted_container.mux(undistorted_video.encode())

		print("Output video saved at: ", undistorted_video_path)
		
if __name__ =='__main__':
# remove the fish eye distortion of the ego video
	undistort_video(sys.argv[1], sys.argv[2])
	# first argument: input video
	# second argument: output video