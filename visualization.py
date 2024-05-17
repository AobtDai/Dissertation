from PIL import Image, ImageDraw, ImageFont

# 创建画布
canvas_width = 800
canvas_height = 800
canvas = Image.new('RGB', (canvas_width, canvas_height), 'white')

original = "./dataset/translated-LLaVA-Instruct-150K/llava-imgs/filtered-llava-images/000000442786.jpg"
GaussianBlur = "./dataset/MyDataset/GaussianBlur/000000442786.jpg"
GaussianNoise = "./dataset/MyDataset/GaussianNoise/000000442786.jpg"
Complex = "./dataset/MyDataset/GaussianBlur/000000442786.jpg"
image_paths = [original, GaussianBlur, GaussianNoise, Complex]
images = [Image.open(path) for path in image_paths]

# 获取单个图片的宽度和高度
image_width = canvas_width // 2
image_height = canvas_height // 2

# 缩放图片以适应画布
scaled_images = [image.resize((image_width, image_height)) for image in images]

# 创建绘图对象
draw = ImageDraw.Draw(canvas)

# 设置字体
# font = ImageFont.truetype('Arial.ttf', 20)

# 在画布上放置图片和编号注释
for i, image in enumerate(scaled_images):
    x = (i % 2) * image_width
    y = (i // 2) * image_height

    # 将图片粘贴到画布上
    canvas.paste(image, (x, y))

    # 添加编号注释
    label = f'Image {i+1}'
    label_width, label_height = draw.textsize(label)
    label_x = x + (image_width - label_width) // 2
    label_y = y + image_height
    draw.text((label_x, label_y), label, fill='black')

# 保存画布为JPG图片
canvas.save('output.jpg', 'JPEG')