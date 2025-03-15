from main import Canvas2DContext
import math

class Ex2d(Canvas2DContext):
    def drawRotateImage(self, im, x, y, width, height, deg, alpha):
        self.save()
        self.globalAlpha *= alpha
        if not deg:
            self.translate(x, y)
            self.rotate(deg * math.pi / 180)
            self.drawImage(im, -width/2, -height/2, width, height)
        else:
            self.drawImage(im, x - width/2, y - height/2, width, height)
        self.restore()
    
    def drawAnchorESRotateImage(self, im, x, y, width, height, deg, alpha):
        self.save()
        self.globalAlpha *= alpha
        if not deg:
            self.translate(x, y)
            self.rotate(deg * math.pi / 180)
            self.drawImage(im, -width/2, -height, width, height)
        else:
            self.drawImage(im, x - width/2, y - height/2, width, height)
        self.restore()
    
    def drawScaleImage(self, im, x, y, width, height, xs, ys):
        x += width / 2
        y += height / 2
        self.save()
        self.translate(x, y)
        self.scale(xs, ys)
        self.drawImage(im, -width/2, -height / 2, width, height)
        self.restore()

    def drawRPEMultipleRotateText(self, text, x, y, deg, fontsize, color, xs, ys):
        self.save()
        self.translate(x, y)
        self.rotate(deg * math.pi / 180)
        self.scale(xs, ys)

        self.fillStyle = color
        self.textAlign = 'center'
        self.textBaseline = 'middle'
        self.font = f'{fontsize}px pgrFont'

        if '\n' in text and RPEVersion >= 150:
            texts = text.split('\n')
            current_y = 0.0
            for currtext in texts:
                if currtext.strip():
                    self.fillText(currtext, 0, current_y)
                    metrics = self.measureText(currtext)
                    line_height = (metrics.actualBoundingBoxAscent + metrics.actualBoundingBoxDescent) * 1.25
                    current_y += line_height
        else:
            self.fillText(text, 0, 0)
        self.restore()
    
    def drawRotateText(self,text, x, y, deg, fontsize, color, xscale, yscale):
        self.save()
        self.translate(x, y)
        self.rotate(deg * math.pi / 180)
        self.scale(xscale, yscale)
        self.fillStyle = color
        self.textAlign = "center"
        self.textBaseline = "middle"
        self.font = f"{fontsize}px pgrFont"
        self.fillText(text, 0, 0)
        self.restore()
    
    def drawAlphaImage(self, im, x, y, width, height, alpha):
        self.save()
        self.globalAlpha *= alpha
        self.drawImage(im, x, y, width, height)
        self.restore()
    
    def drawAlphaCenterImage(self, im, x, y, width, height, alpha):
        self.save()
        self.globalAlpha *= alpha
        self.drawImage(im, x - width / 2, y - height / 2, width, height)
        self.restore()
    
    def drawTextEx(self, text, x, y, font, color, align, baseline):
        self.save()
        self.fillStyle = color
        self.textAlign = align
        self.textBaseline = baseline
        self.font = font
        self.fillText(text, x, y)
        self.restore()
    
    def fillRectEx(self, x, y, w, h, color):
        self.save()
        self.fillStyle = color
        self.fillRect(x, y, w, h)
        self.restore()
    
    def fillRectExConvert2LeftCenter(self, x, y, w, h, color):
        y += h / 2
        self.save()
        self.fillStyle = color
        self.beginPath()
        self.moveTo(x, y - h / 2)
        self.lineTo(x + w, y - h / 2)
        self.lineTo(x + w, y + h / 2)
        self.lineTo(x, y + h / 2)
        self.closePath()
        self.fill()
        self.restore()
    
    def fillRectExByRect(self, x0, y0, x1, y1, color):
        self.fillRectEx(x0, y0, x1 - x0, y1 - y0, color)
    
    def strokeRectEx(self, x, y, w, h, color, width):
        self.save()
        self.strokeStyle = color
        self.lineWidth = width
        self.strokeRect(x, y, w, h)
        self.restore()
    
    def addRoundRectData(self, x, y, w, h, r):
        try:
            if not self._roundDatas:
                self._roundDatas = []
        except:
            self._roundDatas = []
        self._roundDatas.append([x, y, w, h, r])
    
    def drawRoundDatas(self, color):
        if self._roundDatas:
            self.roundRectsEx(color, self._roundDatas)
            self._roundDatas = []
    
    def roundRectsEx(self, color, datas):
        self.save()
        self.fillStyle = color
        self.beginPath()
        for p in datas:
            self.roundRect(*p)
        self.fill()
        self.restore()
    
    def drawLineEx(self, x1, y1, x2, y2, width, color):
        self.save()
        self.strokeStyle = color
        self.lineWidth = width
        self.beginPath()
        self.moveTo(x1, y1)
        self.lineTo(x2, y2)
        self.stroke()
        self.restore()
    
    def _diagonalRectangle(self, x0, y0, x1, y1, power):
        x0 = math.floor(x0)
        y0 = math.floor(y0)
        x1 = math.floor(x1)
        y1 = math.floor(y1)
        self.moveTo(x0 + (x1 - x0) * power, y0)
        self.lineTo(x1, y0)
        self.lineTo(x1 - (x1 - x0) * power, y1)
        self.lineTo(x0, y1)
        self.lineTo(x0 + (x1 - x0) * power, y0)
    
    def clipDiagonalRectangle(self, x0, y0, x1, y1, power):
        self.beginPath()
        self._diagonalRectangle(x0, y0, x1, y1, power)
        self.clip()
    
    def clipRect(self, x0, y0, x1, y1):
        self.beginPath()
        self.rect(x0, y0, x1 - x0, y1 - y0)
        self.clip()
    
    def drawClipXText(self, text, x, y, align, baseline, color, font, clipx0, clipx1):
        self.save()
        self.clipRect(clipx0, 0, clipx1, self.canvas.height)
        self.fillStyle = color
        self.textAlign = align
        self.textBaseline = baseline
        self.font = font
        self.fillText(text, x, y)
        self.restore()
    
    def drawDiagonalRectangle(self, x0, y0, x1, y1, power, color):
        self.save()
        self.fillStyle = color
        self.beginPath()
        self._diagonalRectangle(x0, y0, x1, y1, power)
        self.fill()
        self.restore()
    
    def drawDiagonalRectangleShadow(self, x0, y0, x1, y1, power, color, shadowColor, shadowBlur):
        self.save()
        self.shadowColor = shadowColor
        self.shadowBlur = shadowBlur
        self.fillStyle = color
        self.beginPath()
        self._diagonalRectangle(x0, y0, x1, y1, power)
        self.fill()
        self.restore()
    
    def drawDiagonalDialogRectangleText(self, x0, y0, x1, y1, power, text1, text2, color, font):
        self.save()
        self.fillStyle = color
        self.font = font
        self.textBaseline = "middle"
        self.textAlign = "left"
        self.fillText(text1, x0 + (x1 - x0) * power * 3.0, y0 + (y1 - y0) * 0.5)
        self.textAlign = "right"
        self.fillText(text2, x1 - (x1 - x0) * power * 2.0, y0 + (y1 - y0) * 0.5)
        self.restore()
    
    def drawDiagonalRectangleClipImage(self, x0, y0, x1, y1, im, imx, imy, imw, imh, power, alpha):
        if alpha == 0.0: return
        self.save()
        self.globalAlpha *= alpha
        self.beginPath()
        self._diagonalRectangle(x0, y0, x1, y1, power)
        self.clip()
        self.drawImage(im, x0 + imx, y0 + imy, imw, imh)
        self.restore()
    
    def drawGrd(self, grdpos, steps, x0, y0, x1, y1):
        pass

    def drawDiagonalGrd(self, x0, y0, x1, y1, power, steps, grdpos):
        pass
    
    def drawDiagonalRectangleClipImageOnlyHeight(self, x0, y0, x1, y1, im, imh, power, alpha):
        # 获取图像原始尺寸
        try:
            # 假设im可能有__drawImage__方法返回纹理对象
            if hasattr(im, '__drawImage__'):
                img_obj = im.__drawImage__()
                irw = img_obj.width
                irh = img_obj.height
            else:
                # 直接获取图像尺寸属性
                irw = im.width
                irh = im.height
        except AttributeError:
            # 处理无效图像的情况
            irw = self.width
            irh = self.height

        # 计算目标宽度
        imw = imh * irw / irh

        # 确保宽度足够覆盖区域
        if imw < (x1 - x0):
            imw = x1 - x0
            imh = imw * irh / irw

        # 处理无效尺寸
        if math.isnan(imw) or math.isnan(imh):
            imw = self.width
            imh = self.height

        # 计算图像位置
        imx = (x1 - x0) / 2 - imw / 2
        imy = (y1 - y0) / 2 - imh / 2

        # 调用现有方法进行实际绘制
        return self.drawDiagonalRectangleClipImage(
            x0, y0, x1, y1, im, imx, imy, imw, imh, power, alpha
        )
    
    def drawRotateText2(self, text, x, y, deg, color, font, align, baseline):
        self.save()
        self.translate(x, y)
        self.rotate(deg * math.PI / 180)
        self.fillStyle = color
        self.textAlign = align
        self.textBaseline = baseline
        self.font = font
        self.fillText(text, 0, 0)
        self.restore()
    
    def drawTriangleFrame(self, x0, y0, x1, y1, x2, y2, color, width):
        self.save()
        self.strokeStyle = color
        self.lineWidth = width
        self.beginPath()
        self.moveTo(x0, y0)
        self.lineTo(x1, y1)
        self.lineTo(x2, y2)
        self.closePath()
        self.stroke()
        self.restore()
    
    def drawRectMultilineText(self, x0, y0, x1, y1, text, color, font, fontsize, lineOffsetScale):
        self.save()
        
        # 设置文本属性
        self.font = font
        self.fillStyle = color
        self.textBaseline = "top"
        self.textAlign = "left"

        # 文本换行处理
        max_width = x1 - x0
        lines = []
        current_line = ""
        words = text.split(' ')
        
        for word in words:
            test_line = f"{current_line} {word}".strip()
            # 使用measureText获取文本宽度
            metrics = self.measureText(test_line)
            if metrics.width <= max_width:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)
        
        # 绘制背景矩形
        self.fillRect(x0, y0, x1 - x0, y1 - y0)
        
        # 绘制文本行
        dy = 0.0
        for line in lines:
            self.fillText(line, x0, y0 + dy)
            dy += fontsize * lineOffsetScale
            if dy > (y1 - y0):
                break  # 防止超出区域
        
        self.restore()
        
        return len(lines) * fontsize * lineOffsetScale

    def drawRectMultilineTextDiagonal(self, x0: float, y0: float, x1: float, y1: float, text: str, color: str, font: str, fontsize: float, line_diagonal: float, line_offset_scale: float) -> float:
        self.save()
        
        # 设置文本属性
        self.font = font
        self.fillStyle = color
        self.textBaseline = "top"
        self.textAlign = "left"

        # 计算最大宽度
        max_width = x1 - x0

        # 文本换行处理（模拟splitText逻辑）
        words = text.split(' ')
        lines = []
        current_line = ""
        for word in words:
            test_line = f"{current_line} {word}".strip() if current_line else word
            metrics = self.measureText(test_line)
            if metrics.width <= max_width:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)

        # 绘制背景矩形
        self.fillRect(x0, y0, x1 - x0, y1 - y0)  # 原代码rect可能用于定义剪切区域，这里改为直接填充

        dx = 0.0
        dy = 0.0

        for line in lines:
            if line.strip():  # 非空文本行
                self.fillText(line, x0 + dx, y0 + dy)
                dy += fontsize * line_offset_scale
                dx += line_diagonal
            else:  # 处理空行情况（如换行符）
                dx += line_diagonal * 0.5
                dy += fontsize * line_offset_scale * 0.5
            if dy >= (y1 - y0):
                break  # 防止超出区域

        self.restore()
        return len(lines) * fontsize * line_offset_scale

    def drawRectMultilineTextCenter(self, x0: float, y0: float, x1: float, y1: float, 
                               text: str, color: str, font: str, fontsize: float, 
                               lineOffsetScale: float) -> float:
        # 保存当前绘图状态
        self.save()

        # 设置文本属性
        self.font = font
        self.fillStyle = color
        self.textBaseline = "top"
        self.textAlign = "center"

        # 文本换行处理
        max_width = x1 - x0
        lines = []
        current_line = ""
        words = text.split(' ')
        
        for word in words:
            test_line = f"{current_line} {word}".strip() if current_line else word
            metrics = self.measureText(test_line)
            if metrics.width <= max_width:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)

        # 绘制背景矩形（原rect的剪切区域）
        self.fillRect(x0, y0, x1 - x0, y1 - y0)

        dy = 0.0
        for line in lines:
            # 居中绘制文本行
            self.fillText(line, x0 + (x1 - x0)/2, y0 + dy)
            dy += fontsize * lineOffsetScale
            if dy >= (y1 - y0):
                break  # 防止超出区域

        # 恢复状态
        self.restore()
        
        return len(lines) * fontsize * lineOffsetScale

    def drawUIItems(self, datas):
        for item in datas:
            if not item:
                continue

            if item['type'] == 'text':
                self.save()
                # 处理字体样式
                font_style = f"{item.get('weight', '')} {item['fontsize']}px {item.get('font', 'pgrFont')}"
                self.font = font_style.strip()
                
                # 设置文本属性
                self.textBaseline = item.get('textBaseline', 'alphabetic')
                self.textAlign = item.get('textAlign', 'left')
                self.fillStyle = item['color']
                
                # 变换处理
                self.translate(item['x'] + item.get('dx', 0), item['y'] + item.get('dy', 0))
                if item.get('sx', 1.0) != 1.0 or item.get('sy', 1.0) != 1.0:
                    self.scale(item['sx'], item['sy'])
                if item.get('rotate', 0.0) != 0.0:
                    self.rotate(math.radians(item['rotate']))
                
                # 绘制文本
                self.fillText(item['text'], 0, 0)
                self.restore()
            
            elif item['type'] == 'image':
                self.save()
                # 获取图像对象
                img = item['image']
                
                # 解析颜色参数
                r, g, b, a = item['color']
                self.translate(item['x'] + item.get('dx', 0), item['y'] + item.get('dy', 0))
                if item.get('rotate', 0.0) != 0.0:
                    self.rotate(math.radians(item['rotate']))
                
                # 处理透明度和颜色矩阵
                if a != 1.0:
                    self.globalAlpha = a
                if r != 255 or g != 255 or b != 255:
                    # 在Kivy中可能需要通过Color指令实现颜色矩阵效果
                    self.set_color_matrix(r, g, b)
                    self.filter = "texture_color_filter"
                
                # 绘制图像
                self.drawImage(img, 0, 0, item['width'], item['height'])
                self.restore()
            
            elif item['type'] == 'call':
                method = getattr(self, item['name'], None)
                if method:
                    method(*item['args'])
            
            elif item['type'] == 'pbar':
                self.save()
                w, pw, process = item['w'], item['pw'], item['process']
                
                # 解析颜色参数
                color_str = item['color']
                r, g, b, a = map(float, color_str.split('(')[1].split(')')[0].split(', '))
                
                # 绘制进度条
                fill_color = f"rgba({145*r/255}, {145*g/255}, {145*b/255}, {0.85*a})"
                border_color = f"rgba({r}, {g}, {b}, {0.9*a})"
                
                # 使用Kivy的fillRectExConvert2LeftCenter实现
                self.fillRectExConvert2LeftCenter(
                    0, 0, w * process, pw, fill_color
                )
                self.fillRectExConvert2LeftCenter(
                    w * process - w * 0.00175, 0, w * 0.00175, pw, border_color
                )
                self.restore()
    
    def drawCoverFullScreenImage(self, img, w, h):
        # 获取图像原始尺寸
        if hasattr(img, 'texture'):
            imw_orig, imh_orig = img.texture.size
        else:
            imw_orig, imh_orig = img.size

        # 计算宽高比
        ratio = w / h
        imratio = imw_orig / imh_orig

        # 根据宽高比调整图像尺寸
        if imratio > ratio:
            # 图像更宽，以宽度填满目标区域
            imw = w
            imh = imw / imratio
        else:
            # 图像更高，以高度填满目标区域
            imh = h
            imw = imh * imratio

        # 计算居中位置
        imx = (w - imw) / 2
        imy = (h - imh) / 2

        # 保存当前画布状态
        self.save()

        self.beginPath()
        self.rect(0, 0, w, h)
        self.clip()
        self.drawImage(img, imx, imy, imw, imh)

        # 恢复画布状态
        self.restore()

        return [imx, imy, imw, imh]

    def outOfTransformDrawCoverFullscreenChartBackgroundImage(self, img):
        # 保存当前绘图状态
        self.save()
        
        # 重置所有变换
        self.resetTransform()
        
        # 获取图像居中后的坐标和尺寸
        imx, imy, imw, imh = self.drawCoverFullScreenImage(img, self.width, self.height)
        
        # 设置填充颜色（rgba(0.1, 0.1, 0.1, 0.7)）
        self.fillStyle = (0.1, 0.1, 0.1, 0.7)
        
        # 绘制填充矩形
        self.fillRect(imx, imy, imw, imh)
        
        # 恢复之前保存的绘图状态
        self.restore()
    
    def drawMirrorImage(self, img, x, y, width, height, alpha):
        self.save()
        self.translate(x + width, y)
        self.scale(-1, 1)
        self.globalAlpha = alpha
        self.drawImage(img, 0, 0, width, height)
        self.restore()
    
    def drawMirrorRotateImage(self, img, x, y, width, height, rotate, alpha):
        self.save()
        self.translate(x + width, y)
        self.rotate(rotate * math.PI / 180)
        self.scale(-1, 1)
        self.globalAlpha = alpha
        self.drawImage(img, 0, 0, width, height)
        self.restore()
    
    def getTextSize(self, text, font):
        self.save()
        self.font = font
        measure = self.measureText(text)
        self.restore()
        return [measure.width, measure.actualBoundingBoxAscent + measure.actualBoundingBoxDescent]

    def setShadow(self, color, blur, dx = 0, dy = 0):
        self.save()
        self.shadowColor = color
        self.shadowBlur = blur
        self.shadowOffsetX = dx
        self.shadowOffsetY = dy
    
    def mirror(self):
        self.save()
        self.scale(-1, 1)
        self.translate(-self.width, 0)
        self.drawImage(self.getImageData(0,0, self.width, self.height).texture, 0, 0)
        self.restore()
    
    def clear(self):
        self.save()
        self.setTransform(1, 0, 0, 1, 0, 0)
        self.clearRect(0, 0, self.width, self.height)
        self.restore()

ctx = Ex2d()
RPEVersion = 100
import time
from kivy.app import App
from threading import Thread

class ctxApp(App):
    def build(self, **kwargs):
        ctx.regFont('pgrFont', 'Test/font.ttf')
        background_blur_img = r"C:\Users\Admin\Documents\PhigrosPlayer\src\resources\AllSongBlur.png"
        Note_Hold_End_dub_img = r"C:\Users\Admin\Documents\PhigrosPlayer\src\resources\resource_packs\default\hold_mh.png"
        Note_Hold_Body_dub_img = r"C:\Users\Admin\Documents\PhigrosPlayer\src\resources\resource_packs\default\hold.png"
        Note_Tap_img = r"C:\Users\Admin\Documents\PhigrosPlayer\src\resources\resource_packs\default\click.png"
        Note_Drag_dub_img = r"C:\Users\Admin\Documents\PhigrosPlayer\src\resources\resource_packs\default\drag.png"
        Note_Tap_dub_img = r"C:\Users\Admin\Documents\PhigrosPlayer\src\resources\resource_packs\default\click_mh.png"
        Note_Click_Effect_Perfect_6_img = r"C:\Users\Admin\Documents\PhigrosPlayer\src\resources\resource_packs\default\flick.png"
        background_blur_img = ctx.loadTexture(background_blur_img)
        Note_Hold_End_dub_img = ctx.loadTexture(Note_Hold_End_dub_img)
        Note_Hold_Body_dub_img = ctx.loadTexture(Note_Hold_Body_dub_img)
        Note_Tap_img = ctx.loadTexture(Note_Tap_img)
        Note_Drag_dub_img = ctx.loadTexture(Note_Drag_dub_img)
        Note_Tap_dub_img = ctx.loadTexture(Note_Tap_dub_img)
        Note_Click_Effect_Perfect_6_img = ctx.loadTexture(Note_Click_Effect_Perfect_6_img)
        
        ctx.clear()
        ctx.outOfTransformDrawCoverFullscreenChartBackgroundImage(background_blur_img)
        ctx.save()
        ctx.rect(0, 0, 1152, 648)
        ctx.clip(fill_rule = 'nonzero')
        ctx.drawCoverFullScreenImage(background_blur_img, 1152, 648)
        ctx.fillRectEx(0, 0, 1152, 648, 'rgba(0, 0, 0, 0.6)')
        ctx.drawLineEx(                        2273.317368095187,668.4084405963529,-1121.3173680951866,311.6179526427623,                        3.6,                        'rgba(255, 236, 159, 0.7552536707590608)'                    )
        ctx.drawLineEx(                        2285.18823130074,589.8578553139554,-1109.4465048896332,233.0673673603648,                        3.6,                        'rgba(255, 236, 159, 0.3054205903819961)'                    )
        ctx.drawLineEx(                        600.0070320374591,-1188.1745109523295,243.2165440838685,2206.4602252380437,                        3.6,                        'rgba(255, 236, 159, 0.8655587778371919)'                    )
        ctx.drawRotateImage(Note_Hold_End_dub_img, 599.3712672312153, 302.04357606620454, 146.8974721941355, 13.555510616784632, 6.0, 1.0)
        ctx.drawAnchorESRotateImage(                                    Note_Hold_Body_dub_img,                                    587.8708632055534, 411.4626113371601,                                    146.8974721941355,                                    103.24399059840594,                                    6.0,                                    1.0                                )
        ctx.drawRotateImage(Note_Tap_img, 281.12244709095313, 169.04495380297533, 136.8, 13.832153690596563, 6.0, 1.0)
        ctx.drawRotateImage(Note_Tap_dub_img, 576.0, -46.28571428571426, 146.8974721941355, 26.97841546265115, 0.0, 1.0)
        ctx.drawRotateImage(Note_Drag_dub_img, 576.0, -46.28571428571426, 146.8974721941355, 21.582732370120922, 0.0, 1.0)
        ctx.drawRotateImage(Note_Tap_dub_img, 576.0, -46.28571428571426, 146.8974721941355, 26.97841546265115, 0.0, 1.0)
        ctx.drawRotateImage(Note_Drag_dub_img, 576.0, -46.28571428571426, 146.8974721941355, 21.582732370120922, 0.0, 1.0)
        ctx.drawRotateImage(Note_Tap_dub_img, 576.0, -46.28571428571426, 146.8974721941355, 26.97841546265115, 0.0, 1.0)
        ctx.drawRotateImage(Note_Drag_dub_img, 576.0, -46.28571428571426, 146.8974721941355, 21.582732370120922, 0.0, 1.0)
        ctx.drawRotateImage(Note_Tap_dub_img, 576.0, -46.28571428571426, 146.8974721941355, 26.97841546265115, 0.0, 1.0)
        ctx.drawRotateImage(Note_Drag_dub_img, 576.0, -46.28571428571426, 146.8974721941355, 21.582732370120922, 0.0, 1.0)
        ctx.drawRotateImage(Note_Tap_dub_img, 576.0, -46.28571428571426, 146.8974721941355, 26.97841546265115, 0.0, 1.0)
        ctx.drawRotateImage(Note_Drag_dub_img, 576.0, -46.28571428571426, 146.8974721941355, 21.582732370120922, 0.0, 1.0)
        ctx.drawRotateImage(Note_Tap_dub_img, 576.0, -46.28571428571426, 146.8974721941355, 26.97841546265115, 0.0, 1.0)
        ctx.drawRotateImage(Note_Drag_dub_img, 576.0, -46.28571428571426, 146.8974721941355, 21.582732370120922, 0.0, 1.0)
        ctx.drawRotateImage(Note_Tap_dub_img, 576.0, -46.28571428571426, 146.8974721941355, 26.97841546265115, 0.0, 1.0)
        ctx.drawRotateImage(Note_Drag_dub_img, 576.0, -46.28571428571426, 146.8974721941355, 21.582732370120922, 0.0, 1.0)
        ctx.drawRotateImage(Note_Tap_dub_img, 576.0, -46.28571428571426, 146.8974721941355, 26.97841546265115, 0.0, 1.0)
        ctx.drawRotateImage(Note_Drag_dub_img, 576.0, -46.28571428571426, 146.8974721941355, 21.582732370120922, 0.0, 1.0)
        ctx.drawRotateImage(Note_Drag_dub_img, 576.0, -46.28571428571426, 146.8974721941355, 21.582732370120922, 0.0, 1.0)
        ctx.drawRotateImage(Note_Tap_dub_img, 576.0, -46.28571428571426, 146.8974721941355, 26.97841546265115, 0.0, 1.0)
        ctx.drawRotateImage(Note_Drag_dub_img, 576.0, -46.28571428571426, 146.8974721941355, 21.582732370120922, 0.0, 1.0)
        ctx.drawRotateImage(Note_Tap_dub_img, 576.0, -46.28571428571426, 146.8974721941355, 26.97841546265115, 0.0, 1.0)
        ctx.drawRotateImage(Note_Drag_dub_img, 576.0, -46.28571428571426, 146.8974721941355, 21.582732370120922, 0.0, 1.0)
        ctx.addRoundRectData(541.4905685395674, 604.6798768090285, 20.83666922510153, 20.83666922510153, 0.0)
        ctx.addRoundRectData(492.3940445157255, 570.5236001450817, 20.83666922510153, 20.83666922510153, 0.0)
        ctx.addRoundRectData(671.7961601573045, 490.90346876095657, 20.83666922510153, 20.83666922510153, 0.0)
        ctx.addRoundRectData(674.7920595898686, 488.25425294565565, 20.83666922510153, 20.83666922510153, 0.0)
        ctx.drawRoundDatas('rgba(255, 236, 160, 0.8033977826436356)')
        ctx.drawAlphaImage(Note_Click_Effect_Perfect_6_img, 470.664, 403.8068571428571, 210.67200000000005, 210.67200000000005, 0.8823529411764706)
        ctx.addRoundRectData(546.6698609250883, 605.7603189578796, 20.83666922510153, 20.83666922510153, 0.0)
        ctx.addRoundRectData(470.7705962792951, 526.761832812409, 20.83666922510153, 20.83666922510153, 0.0)
        ctx.addRoundRectData(662.4305789268783, 484.66113983014947, 20.83666922510153, 20.83666922510153, 0.0)
        ctx.addRoundRectData(493.48997389601743, 564.298397387348, 20.83666922510153, 20.83666922510153, 0.0)
        ctx.drawRoundDatas('rgba(255, 236, 160, 0.8033977826436356)')
        ctx.drawAlphaImage(Note_Click_Effect_Perfect_6_img, 470.664, 403.8068571428571, 210.67200000000005, 210.67200000000005, 0.8823529411764706)
        ctx.drawUIItems([None, {'type': 'pbar', 'w': 1152, 'pw': 6.821052631578947, 'process': 0.011127999496295591, 'dx': 0.0, 'dy': 0.0, 'sx': 1.0, 'sy': 1.0, 'color': 'rgba(255, 255, 255, 1.0)', 'rotate': 0.0}, {'type': 'text', 'text': '0004375', 'fontsize': 32.0, 'textBaseline': 'top', 'textAlign': 'right', 'x': 1128.0, 'y': 18.6, 'dx': 0.0, 'dy': 0.0, 'sx': 1.0, 'sy': 1.0, 'color': 'rgba(255, 255, 255, 1.0)', 'rotate': 0.0}, None, {'type': 'text', 'text': '7', 'fontsize': 45.28301886792453, 'textBaseline': 'middle', 'textAlign': 'center', 'x': 576.0, 'y': 31.2, 'dx': 0.0, 'dy': 0.0, 'sx': 1.0, 'sy': 1.0, 'color': 'rgba(255, 255, 255, 1.0)', 'rotate': 0.0}, {'type': 'text', 'text': 'AUTOPLAY', 'fontsize': 15.0, 'textBaseline': 'top', 'textAlign': 'center', 'x': 576.0, 'y': 55.080000000000005, 'dx': 0.0, 'dy': 0.0, 'sx': 1.0, 'sy': 1.0, 'color': 'rgba(255, 255, 255, 1.0)', 'rotate': 0.0}, {'type': 'image', 'image': r"C:\Users\Admin\Documents\PhigrosPlayer\src\resources\Pause.png", 'x': 21.599999999999998, 'y': 24.599999999999998, 'dx': 0.0, 'dy': 0.0, 'width': 19.2, 'height': 22.49142857142857, 'rotate': 0.0, 'color': [255.0, 255.0, 255.0, 1.0]}, None, {'type': 'text', 'text': '群青', 'fontsize': 20.869565217391305, 'textBaseline': 'bottom', 'textAlign': 'left', 'x': 25.919999999999998, 'y': 625.3199999999999, 'dx': 0.0, 'dy': 0.0, 'sx': 1.0, 'sy': 1.0, 'color': 'rgba(255, 255, 255, 1.0)', 'rotate': 0.0}, {'type': 'text', 'text': 'Air Lv.A', 'fontsize': 20.869565217391305, 'textBaseline': 'bottom', 'textAlign': 'right', 'x': 1126.08, 'y': 625.3199999999999, 'dx': 0.0, 'dy': 0.0, 'sx': 1.0, 'sy': 1.0, 'color': 'rgba(255, 255, 255, 1.0)', 'rotate': 0.0}, {'type': 'text', 'text': 'fps 65 - reqaf fps 65', 'fontsize': 8.727272727272728, 'textBaseline': 'bottom', 'textAlign': 'center', 'x': 576.0, 'y': 631.8, 'dx': 0.0, 'dy': 0.0, 'sx': 1.0, 'sy': 1.0, 'color': 'rgba(255, 255, 255, 0.5)', 'rotate': 0.0}, {'type': 'text', 'text': 'PhigrosPlayer - by qaqFei - github.com/qaqFei/PhigrosPlayer - MIT License', 'fontsize': 8.727272727272728, 'textBaseline': 'bottom', 'textAlign': 'center', 'x': 576.0, 'y': 641.52, 'dx': 0.0, 'dy': 0.0, 'sx': 1.0, 'sy': 1.0, 'color': 'rgba(255, 255, 255, 0.5)', 'rotate': 0.0}, None, None])
        ctx.restore()
        return ctx

app = ctxApp()

def draw():
    with ctx:
        #ico = ctx.loadTexture('Test/icon.ico')
        ctx.regFont('pgrFont', 'Test/font.ttf')
    while True:
        with ctx:
            pass

        
        time.sleep(1 / 60)


#Thread(target = draw, daemon = True).start()
app.run()