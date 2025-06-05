system_prompt_alignment_en="""
You are a professional document analysis assistant. You can accurately perform text recognition, as well as state the positions of each image and paragraph in the document."""

system_prompt_alignment_zh="""你是一位专业的文档分析助手。你擅长文本识别，以及标注文档中每张图片和每个段落的位置。
"""

system_prompt_recognition_en="""
You are a professional ducument analysis assistant. You can accurately perform text recognition."""

system_prompt_recognition_zh="""
你是一位专业的文档分析助手。你擅长准确地进行文本识别。"""

prompt_alignment_en="""
Below is an image of a document. Analyse the document using Qwen-VL HTML, then extract each figure with its corresponding caption.
Note that there may be multi-panel images. You should add the description of the large figure to the beginning of each sub-figure caption.
For example, if the caption is like this: "Fig. 1.2 Some description. A Sub-figure a description. B Sub-figure b description."
You should output your answer in the following JSON format without any additional explanation:
[
    {
        "bbox": [x1, y1, x2, y2],  # The bounding box of the figure (top-left x1, y1 and bottom-right x2, y2). This comment is not part of the JSON.
        "caption": "Some description. Sub-figure a description."
    },
    {
        "bbox": [x1, y1, x2, y2],
        "caption": "some description. Sub-figure b description."
    }
]
Note that you should remove the figure labels like "Fig. 1.2" and sub-figure labels like "A" or "(a)" in your output.
Note that if there is no caption, you should output exactly: "None".
Note that if the image doesn't look like a document, but a book cover, or a table of content, or a reference list, or a pure figure, or a chart and so on, you should also output exactly: "None".
Note that you should never make up any information that is not in the image.
"""

prompt_alignment_zh="""
以下是一张文档的图片。使用Qwen-VL HTML分析文档，然后提取每个插图及其相应的标题。你需要注意，有些图片可能是复合图，即由好几张子图组合而成。你提取子图标题时应该将复合图的整体描述添加到每个子图标题开头。
例如，如果标题是这样的：“图1.2 大图描述。A 子图a描述。B 子图b描述。”，你应该以以下的JSON格式输出答案，不要添加任何额外的解释说明：
[
    {
        "bbox": [x1, y1, x2, y2],  # 图片的边界框（左上角x1, y1和右下角x2, y2）。这个注释不是JSON的一部分。
        "caption": "大图描述。子图a描述。"
    },
    {
        "bbox": [x1, y1, x2, y2],
        "caption": "大图描述。子图b描述。"
    }
]
注意，在你的输出中应该删除类似“图1.2”的图片标签和类似“A”或“(a)”这样的子图标签。
注意，如果这一页中没有图片或对应的标题，你应该准确输出：“None”这四个字母。
注意，如果这张图片看起来并不像一个文档，而是书的封面、目录、参考文献列表、纯图片、纯图表等，你也应该输出：“None”。
注意，你的输出必须来自图片原文，不能编造图中不存在的任何信息。
"""

prompt_recognition_en="""
Below is an image of a document. Analyse the document using Qwen-VL HTML, first determine which parts of the text in the document are image captions, and which are main text, then output ONLY THE MAIN BODY TEXT (excluding figure captions) in the order it should be read in markdown format, retaining the original hierarchical structure.
You should output the answer without any additional explanation.
Note that if the document image doesn't look like a document, but a book cover, or a table of content, or a reference list, or a pure figure, or a chart and so on, you should also output exactly: "None".
"""

prompt_recognition_zh="""
以下是一张文档的图片。使用Qwen-VL HTML分析文档，首先确定文档中哪些文本部分是图片标题，哪些是正文，然后仅仅以Markdown格式、按照阅读顺序输出正文文本（注意只有正文文本，不包括图片标题），保留原始的层级结构。
你应该直接输出答案，不要添加任何额外说明。
注意，如果这张图片看起来并不像一个文档，而是书的封面、目录、参考文献列表、纯图片、纯图表等，你应该输出：“None”。
"""


system_prompt_inline_en = """You are a reading comprehension master. You can identify relevant contents in the main text that refer or relate to the figure captions."""

system_prompt_inline_zh = """
你是一位阅读理解大师，能够识别正文文本中与某些图片标题相关的内容，或是正文中对图片标题的引用。"""

prompt_inline_en = """Please analyze each figure caption in the following JSON, base on each caption text, identify relevant references from the given main text.
the JSON file is as follows:
{json_text}

You need to find the in-line references in the main text, and add a new key "inline_ref" to each figure caption in the JSON file.
Note that if none of the texts in the main text refer to the figure caption, you should set the value of "inline_ref" to an empty string like this: "".
Your output should be a JSON file with the same structure as the input JSON, but with the "inline_ref" key added to each figure caption.
You should not add any additional explanations.
The main text is as follows:
{main_text}
"""

prompt_inline_zh = """
请分析以下JSON中的每个图片标题所写的内容。对于每个标题文本，你需要从给定的正文文本中识别出与这张图相关的内容或引用。
JSON文件如下：
{json_text}

你需要在正文文本中找到与图片标题相关的内容，并为JSON文件中的每个图片标题添加一个新的键"inline_ref"。
请注意，如果正文文本中没有任何内容与图片标题相关，则应将"inline_ref"的值设置为空字符串，如下所示：""
你的输出应该是一个JSON文件，其结构与输入JSON相同，但每个图片标题都添加了"inline_ref"键。
不需要添加任何额外的解释，直接输出你的答案。
正文文本如下：
{main_text}
"""

language = "zh"