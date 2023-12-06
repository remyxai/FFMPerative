MAIN_PROMPT = 'I will ask you to perform a task, your job is to come up with a series of simple commands in Python that will perform the task.\nTo help you, I will give you access to a set of tools that you can use. Each tool is a Python function and has a description explaining the task it performs, the inputs it expects and the outputs it returns.\nYou should first explain which tool you will use to perform the task and for what reason, then write the code in Python.\nEach instruction in Python should be a simple assignment. You can print intermediate results if it makes sense to do so.\n\nTools:\n- DocumentQa: This is a tool that answers a question about an document (pdf). It takes an input named `document` which should be the document containing the information, as well as a `question` that is the question about the document. It returns a text that contains the answer to the question.\n- ImageCaptioner: This is a tool that generates a description of an image. It takes an input named `image` which should be the image to caption, and returns a text that contains the description in English.\n- ImageQa: This is a tool that answers a question about an image. It takes an input named `image` which should be the image containing the information, as well as a `question` which should be the question in English. It returns a text that is the answer to the question.\n- ImageSegmenter: This is a tool that creates a segmentation mask of an image according to a label. It cannot create an image. It takes two arguments named `image` which should be the original image, and `label` which should be a text describing the elements what should be identified in the segmentation mask. The tool returns the mask.\n- Transcriber: This is a tool that transcribes an audio into text. It takes an input named `audio` and returns the transcribed text.\n- Summarizer: This is a tool that summarizes an English text. It takes an input `text` containing the text to summarize, and returns a summary of the text.\n- TextClassifier: This is a tool that classifies an English text using provided labels. It takes two inputs: `text`, which should be the text to classify, and `labels`, which should be the list of labels to use for classification. It returns the most likely label in the list of provided `labels` for the input text.\n- TextQa: This is a tool that answers questions related to a text. It takes two arguments named `text`, which is the text where to find the answer, and `question`, which is the question, and returns the answer to the question.\n- TextReader: This is a tool that reads an English text out loud. It takes an input named `text` which should contain the text to read (in English) and returns a waveform object containing the sound.\n- Translator: This is a tool that translates text from a language to another. It takes three inputs: `text`, which should be the text to translate, `src_lang`, which should be the language of the text to translate and `tgt_lang`, which should be the language for the desired ouput language. Both `src_lang` and `tgt_lang` are written in plain English, such as \'Romanian\', or \'Albanian\'. It returns the text translated in `tgt_lang`.\n- ImageTransformer: This is a tool that transforms an image according to a prompt. It takes two inputs: `image`, which should be the image to transform, and `prompt`, which should be the prompt to use to change it. The prompt should only contain descriptive adjectives, as if completing the prompt of the original image. It returns the modified image.\n- TextDownloader: This is a tool that downloads a file from a `url`. It takes the `url` as input, and returns the text contained in the file.\n- ImageGenerator: This is a tool that creates an image according to a prompt, which is a text description. It takes an input named `prompt` which contains the image description and outputs an image.\n- VideoGenerator: This is a tool that creates a video according to a text description. It takes an input named `prompt` which contains the image description, as well as an optional input `seconds` which will be the duration of the video. The default is of two seconds. The tool outputs a video object.\n- n\nTools:\n- AudioAdjustmentTool: \n    This tool modifies audio levels for an input video.\n    Inputs are input_path, output_path, level (e.g. 0.5 or -13dB).\n    Output is the output_path.\n    \n- AudioVideoMuxTool: \n    This tool muxes (combines) a video and an audio file.\n    Inputs are input_path as a string, audio_path as a string, and output_path as a string.\n    Output is the output_path.\n    \n- FFProbeTool: \n    This tool extracts metadata from input video using ffmpeg/ffprobe\n    Input is input_path and output is video metadata as JSON.\n    \n- ImageDirectoryToVideoTool: \n    This tool creates video\n    from a directory of images. Inputs\n    are input_path and output_path. \n    Output is the output_path.\n    \n- ImageToVideoTool: \n    This tool generates an N-second video clip from an image.\n    Inputs are image_path, duration, output_path.\n    \n- VideoCropTool: \n    This tool crops a video with inputs: \n    input_path, output_path, \n    top_x, top_y, \n    bottom_x, bottom_y.\n    Output is the output_path.\n    \n- VideoFlipTool: \n    This tool flips video along the horizontal \n    or vertical axis. Inputs are input_path, \n    output_path and orientation. Output is output_path.\n    \n- VideoFrameSampleTool: \n    This tool samples an image frame from an input video. \n    Inputs are input_path, output_path, and frame_number.\n    Output is the output_path.\n    \n- VideoGopChunkerTool: \n    This tool segments video input into GOPs (Group of Pictures) chunks of \n    segment_length (in seconds). Inputs are input_path and segment_length.\n    \n- VideoHTTPServerTool: \n    This tool streams a source video to an HTTP server. \n    Inputs are input_path and server_url.\n    \n- VideoLetterBoxingTool: \n    This tool adds letterboxing to a video.\n    Inputs are input_path, output_path, width, height, bg_color.\n    \n- VideoOverlayTool: \n    This tool overlays one video on top of another.\n    Inputs are main_video_path, overlay_video_path, output_path, x_position, y_position.\n    \n- VideoResizeTool: \n    This tool resizes the video to the specified dimensions.\n    Inputs are input_path, width, height, output_path.\n    \n- VideoReverseTool: \n    This tool reverses a video. \n    Inputs are input_path and output_path.\n    \n- VideoRotateTool: \n    This tool rotates a video by a specified angle. \n    Inputs are input_path, output_path and rotation_angle in degrees.\n    \n- VideoSegmentDeleteTool: \n    This tool deletes a interval of video by timestamp.\n    Inputs are input_path, output_path, start, end.\n    Format start/end as float.\n    \n- VideoSpeedTool: \n    This tool speeds up a video. \n    Inputs are input_path as a string, output_path as a string, speed_factor (float) as a string.\n    Output is the output_path.\n    \n- VideoStackTool: \n    This tool stacks two videos either vertically or horizontally based on the orientation parameter.\n    Inputs are input_path, second_input, output_path, and orientation as strings.\n    Output is the output_path.\n    vertical orientation -> vstack, horizontal orientation -> hstack\n    \n- VideoTrimTool: \n    This tool trims a video. Inputs are input_path, output_path, \n    start_time, and end_time. Format start(end)_time: HH:MM:SS\n    \n- VideoWatermarkTool: \n    This tool adds logo image as watermark to a video. \n    Inputs are input_path, output_path, watermark_path.\n  \n\n\nTask: "Answer the question in the variable `question` about the image stored in the variable `image`. The question is in French."\n\nI will use the following tools: `Translator` to translate the question into English and then `ImageQa` to answer the question on the input image.\n\nAnswer:\n```py\ntranslated_question = translator(question=question, src_lang="French", tgt_lang="English")\nprint(f"The translated question is {translated_question}.")\nanswer = ImageQa(image=image, question=translated_question)\nprint(f"The answer is {answer}")\n```\n\nTask: "Identify the oldest person in the `document` and create an image showcasing the result."\n\nI will use the following tools: `DocumentQa` to find the oldest person in the document, then `ImageGenerator` to generate an image according to the answer.\n\nAnswer:\n```py\nanswer = DocumentQa(document, question="What is the oldest person?")\nprint(f"The answer is {answer}.")\nimage = ImageGenerator(answer)\n```\n\nTask: "Generate an image using the text given in the variable `caption`."\n\nI will use the following tool: `ImageGenerator` to generate an image.\n\nAnswer:\n```py\nimage = ImageGenerator(prompt=caption)\n```\n\nTask: "Summarize the text given in the variable `text` and read it out loud."\n\nI will use the following tools: `Summarizer` to create a summary of the input text, then `TextReader` to read it out loud.\n\nAnswer:\n```py\nsummarized_text = Summarizer(text)\nprint(f"Summary: {summarized_text}")\naudio_summary = TextReader(summarized_text)\n```\n\nTask: "Answer the question in the variable `question` about the text in the variable `text`. Use the answer to generate an image."\n\nI will use the following tools: `TextQa` to create the answer, then `ImageGenerator` to generate an image according to the answer.\n\nAnswer:\n```py\nanswer = TextQa(text=text, question=question)\nprint(f"The answer is {answer}.")\nimage = ImageGenerator(answer)\n```\n\nTask: "Caption the following `image`."\n\nI will use the following tool: `ImageCaptioner` to generate a caption for the image.\n\nAnswer:\n```py\ncaption = ImageCaptioner(image)\n```\n\nTask: "<<prompt>>"\n\nI will use the following'