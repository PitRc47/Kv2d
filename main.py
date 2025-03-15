from kivy.app import App
from kivy.uix.widget import Widget
from kivy.graphics.transformation import Matrix
from kivy.core.window import Window
from kivy.core.text import Label as CoreLabel, LabelBase
from kivy.core.image import Image as CoreImage
from kivy.clock import Clock
from kivy.graphics.fbo import Fbo
from kivy.graphics.texture import Texture
from kivy.graphics.stencil_instructions import StencilPush, StencilPop, StencilUse
from kivy.graphics.opengl import (glReadPixels, 
    glFinish, glFlush, GL_RGBA, GL_UNSIGNED_BYTE
)
from kivy.graphics import (
    RoundedRectangle, Color, Rectangle, Line, ClearColor,
    Mesh, PushMatrix, PopMatrix, ClearBuffers,
    Scale, Translate, MatrixInstruction
)

import re, copy
from functools import wraps

class CSSColorParser:
    COLORS = {
        'aliceblue': (240, 248, 255),
        'antiquewhite': (250, 235, 215),
        'aqua': (0, 255, 255),
        'aquamarine': (127, 255, 212),
        'azure': (240, 255, 255),
        'beige': (245, 245, 220),
        'bisque': (255, 228, 196),
        'black': (0, 0, 0),
        'blanchedalmond': (255, 235, 205),
        'blue': (0, 0, 255),
        'blueviolet': (138, 43, 226),
        'brown': (165, 42, 42),
        'burlywood': (222, 184, 135),
        'cadetblue': (95, 158, 160),
        'chartreuse': (127, 255, 0),
        'chocolate': (210, 105, 30),
        'coral': (255, 127, 80),
        'cornflowerblue': (100, 149, 237),
        'cornsilk': (255, 248, 220),
        'crimson': (220, 20, 60),
        'cyan': (0, 255, 255),
        'darkblue': (0, 0, 139),
        'darkcyan': (0, 139, 139),
        'darkgoldenrod': (184, 134, 11),
        'darkgray': (169, 169, 169),
        'darkgrey': (169, 169, 169),
        'darkgreen': (0, 100, 0),
        'darkkhaki': (189, 183, 107),
        'darkmagenta': (139, 0, 139),
        'darkolivegreen': (85, 107, 47),
        'darkorange': (255, 140, 0),
        'darkorchid': (153, 50, 204),
        'darkred': (139, 0, 0),
        'darksalmon': (233, 150, 122),
        'darkseagreen': (143, 188, 143),
        'darkslateblue': (72, 61, 139),
        'darkslategray': (47, 79, 79),
        'darkslategrey': (47, 79, 79),
        'darkturquoise': (0, 206, 209),
        'darkviolet': (148, 0, 211),
        'deeppink': (255, 20, 147),
        'deepskyblue': (0, 191, 255),
        'dimgray': (105, 105, 105),
        'dimgrey': (105, 105, 105),
        'dodgerblue': (30, 144, 255),
        'firebrick': (178, 34, 34),
        'floralwhite': (255, 250, 240),
        'forestgreen': (34, 139, 34),
        'fuchsia': (255, 0, 255),
        'gainsboro': (220, 220, 220),
        'ghostwhite': (248, 248, 255),
        'gold': (255, 215, 0),
        'goldenrod': (218, 165, 32),
        'gray': (128, 128, 128),
        'grey': (128, 128, 128),
        'green': (0, 128, 0),
        'greenyellow': (173, 255, 47),
        'honeydew': (240, 255, 240),
        'hotpink': (255, 105, 180),
        'indianred': (205, 92, 92),
        'indigo': (75, 0, 130),
        'ivory': (255, 255, 240),
        'khaki': (240, 230, 140),
        'lavender': (230, 230, 250),
        'lavenderblush': (255, 240, 245),
        'lawngreen': (124, 252, 0),
        'lemonchiffon': (255, 250, 205),
        'lightblue': (173, 216, 230),
        'lightcoral': (240, 128, 128),
        'lightcyan': (224, 255, 255),
        'lightgoldenrodyellow': (250, 250, 210),
        'lightgray': (211, 211, 211),
        'lightgrey': (211, 211, 211),
        'lightgreen': (144, 238, 144),
        'lightpink': (255, 182, 193),
        'lightsalmon': (255, 160, 122),
        'lightseagreen': (32, 178, 170),
        'lightskyblue': (135, 206, 250),
        'lightslategray': (119, 136, 153),
        'lightslategrey': (119, 136, 153),
        'lightsteelblue': (176, 196, 222),
        'lightyellow': (255, 255, 224),
        'lime': (0, 255, 0),
        'limegreen': (50, 205, 50),
        'linen': (250, 240, 230),
        'magenta': (255, 0, 255),
        'maroon': (128, 0, 0),
        'mediumaquamarine': (102, 205, 170),
        'mediumblue': (0, 0, 205),
        'mediumorchid': (186, 85, 211),
        'mediumpurple': (147, 112, 219),
        'mediumseagreen': (60, 179, 113),
        'mediumslateblue': (123, 104, 238),
        'mediumspringgreen': (0, 250, 154),
        'mediumturquoise': (72, 209, 204),
        'mediumvioletred': (199, 21, 133),
        'midnightblue': (25, 25, 112),
        'mintcream': (245, 255, 250),
        'mistyrose': (255, 228, 225),
        'moccasin': (255, 228, 181),
        'navajowhite': (255, 222, 173),
        'navy': (0, 0, 128),
        'oldlace': (253, 245, 230),
        'olive': (128, 128, 0),
        'olivedrab': (107, 142, 35),
        'orange': (255, 165, 0),
        'orangered': (255, 69, 0),
        'orchid': (218, 112, 214),
        'palegoldenrod': (238, 232, 170),
        'palegreen': (152, 251, 152),
        'paleturquoise': (175, 238, 238),
        'palevioletred': (219, 112, 147),
        'papayawhip': (255, 239, 213),
        'peachpuff': (255, 218, 185),
        'peru': (205, 133, 63),
        'pink': (255, 192, 203),
        'plum': (221, 160, 221),
        'powderblue': (176, 224, 230),
        'purple': (128, 0, 128),
        'rebeccapurple': (102, 51, 153),
        'red': (255, 0, 0),
        'rosybrown': (188, 143, 143),
        'royalblue': (65, 105, 225),
        'saddlebrown': (139, 69, 19),
        'salmon': (250, 128, 114),
        'sandybrown': (244, 164, 96),
        'seagreen': (46, 139, 87),
        'seashell': (255, 245, 238),
        'sienna': (160, 82, 45),
        'silver': (192, 192, 192),
        'skyblue': (135, 206, 235),
        'slateblue': (106, 90, 205),
        'slategray': (112, 128, 144),
        'slategrey': (112, 128, 144),
        'snow': (255, 250, 250),
        'springgreen': (0, 255, 127),
        'steelblue': (70, 130, 180),
        'tan': (210, 180, 140),
        'teal': (0, 128, 128),
        'thistle': (216, 191, 216),
        'tomato': (255, 99, 71),
        'turquoise': (64, 224, 208),
        'violet': (238, 130, 238),
        'wheat': (245, 222, 179),
        'white': (255, 255, 255),
        'whitesmoke': (245, 245, 245),
        'yellow': (255, 255, 0),
        'yellowgreen': (154, 205, 50)
    }

    @classmethod
    def parse_color(cls, color_str):
        color_str = color_str.strip().lower()
        if not color_str:
            raise ValueError("Empty color string")

        if color_str in cls.COLORS:
            r, g, b = cls.COLORS[color_str]
            return (r/255.0, g/255.0, b/255.0, 1.0)

        if color_str.startswith('#'):
            return cls._parse_hex(color_str)

        if color_str.startswith(('rgb', 'rgba')):
            return cls._parse_rgb(color_str)

        raise ValueError(f"Unrecognized color format: {color_str}")

    @classmethod
    def _parse_hex(cls, color_str):
        hex_str = color_str.lstrip('#')
        length = len(hex_str)

        if length not in (3, 4, 6, 8):
            raise ValueError(f"Invalid hex color: {color_str}")

        if length in (3, 4):
            hex_str = ''.join([c*2 for c in hex_str])
            length = len(hex_str)

        try:
            components = [int(hex_str[i:i+2], 16) for i in range(0, length, 2)]
            r = components[0] / 255.0
            g = components[1] / 255.0
            b = components[2] / 255.0
            a = components[3]/255.0 if length ==8 else 1.0
        except ValueError:
            raise ValueError(f"Invalid hex color: {color_str}")

        return (r, g, b, a)

    @classmethod
    def _parse_rgb(cls, color_str):
        match = re.match(r'^rgba?\((.*)\)$', color_str, re.IGNORECASE)
        if not match:
            raise ValueError(f"Invalid RGB format: {color_str}")

        components = []
        for part in re.split(r'[,\s/]+', match.group(1)):
            part = part.strip()
            if part:
                components.append(part)

        if len(components) not in (3, 4):
            raise ValueError(f"Invalid RGB components: {color_str}")

        r = cls._parse_component(components[0], max_val=255)
        g = cls._parse_component(components[1], max_val=255)
        b = cls._parse_component(components[2], max_val=255)
        a = cls._parse_component(components[3], max_val=1.0) if len(components)>3 else 1.0

        return (r, g, b, a)

    @classmethod
    def _parse_component(cls, component, max_val):
        component = component.strip().lower()
        if component.endswith('%'):
            value = float(component[:-1]) * max_val / 100.0
        else:
            value = float(component)
        
        normalized = value / (255.0 if max_val ==255 else 1.0)
        return max(0.0, min(1.0, normalized))

class CSSFont:
    def __init__(self, font_str):
        self.font_str = font_str
        self._parse_font_str(font_str)
    
    def _parse_font_str(self, font_str):
        pattern = re.compile(r"""
            ^
            (?:
                (?:italic|oblique|normal)(?:\s+(?:italic|oblique|normal))*\s+
            )?
            (?:
                (?:small-caps|normal)(?:\s+(?:small-caps|normal))*\s+
            )?
            (?:
                (?:bold|lighter|bolder|\d{3}|normal)(?:\s+(?:bold|lighter|bolder|\d{3}|normal))*\s+
            )?
            (\d+\.?\d*)
            (px|pt|em|mm)?
            (?:
                \s*/\s*
                (\d+\.?\d*)
                (px|pt|em|mm)?
                \s*
            )?
            (?:
                \s*/\s*
                (\d+\.?\d*)
                (px|pt|em|mm)?
                \s*
            )?
            \s+
            (
                (?:'[^']*'|"[^"]*"|\w+(?:\s+\w+)*)
                (?:
                    \s*,\s*
                    (?:'[^']*'|"[^"]*"|\w+(?:\s+\w+)*)  
                )*
            )
            $
        """, re.VERBOSE | re.IGNORECASE)

        match = pattern.match(font_str.strip())
        if not match:
            raise ValueError(f"Invalid font format: {font_str}")

        self.font_style = match.group(1) or 'normal'
        self.font_variant = match.group(2) or 'normal'
        self.font_weight = match.group(3) or 'normal'
        self._parse_size(match)
        self._parse_font_family(match.group(7))

    def _parse_size(self, match):
        size_val = float(match.group(1))
        size_unit = match.group(2) or 'px'
        self.font_size = self._convert_unit(size_val, size_unit)

        if match.group(3):
            line_height_val = float(match.group(3))
            line_height_unit = match.group(4) or 'px'
            self.line_height = self._convert_unit(line_height_val, line_height_unit)
        else:
            self.line_height = None

    def _convert_unit(self, value, unit):
        conversions = {
            'px': lambda x: x,
            'pt': lambda x: x * 1.3333,
            'em': lambda x: x * 16,
            'mm': lambda x: x * 3.7795
        }
        return conversions[unit.lower()](value) if unit in conversions else value

    def _parse_font_family(self, family_str):
        families = []
        current = []
        in_quote = False
        quote_char = None
        
        for c in family_str.strip():
            if c in ('"', "'"):
                if not in_quote:
                    in_quote = True
                    quote_char = c
                elif c == quote_char:
                    in_quote = False
                    quote_char = None
                continue
            
            if not in_quote and c == ',':
                if current:
                    families.append(''.join(current).strip())
                    current = []
                continue
                
            current.append(c)
        
        if current:
            families.append(''.join(current).strip())
        
        self.font_family = families

    def apply_to_text(self, text_widget):
        if self.font_size:
            text_widget._font_size = self.font_size
        if self.font_family:
            text_widget._font_name = self._get_kivy_font_name()
        self._apply_font_style(text_widget)
        self._apply_font_weight(text_widget)

    def _get_kivy_font_name(self):
        kivy_fonts = []
        for font in self.font_family:
            font = font.strip('\'"')
            kivy_fonts.append(font)
        return ','.join(kivy_fonts)

    def _apply_font_style(self, widget):
        if 'italic' in self.font_style.lower() or 'oblique' in self.font_style.lower():
            widget.font_style = 'italic'

    def _apply_font_weight(self, widget):
        weight_map = {'bold': '700', 'normal': '400', 'lighter': '300', 'bolder': '700'}
        weight = self.font_weight.lower()
        weight = weight_map.get(weight, weight)
        if weight.isdigit() and int(weight) >= 600:
            widget.bold = True

class TextMetrics:
    def __init__(self, label, context):
        self._label = label
        self._ctx = context
        self._texture = label.texture if label else None
        self._extents = label.get_extents(label.text) if label.text else None

    @property
    def width(self) -> float:
        return self._extents[0] if self._extents else 0

    @property
    def ascent(self):
        return self.font_height * 0.8

    @property
    def descent(self):
        return self.font_height * 0.2

    @property
    def font_height(self):
        return self._extents[1] if self._extents else 0

    @property
    def actualBoundingBoxAscent(self) -> float:
        return self.ascent

    @property
    def actualBoundingBoxDescent(self) -> float:
        return self.descent

    @property
    def alphabeticBaseline(self) -> float:
        return self.ascent

    @property
    def ideographicBaseline(self) -> float:
        return self.font_height * 0.9
    
    @property
    def hangingBaseline(self):
        return self.ascent * 0.2

    def _get_alignment_base_x(self):
        align = self._ctx.textAlign
        if align == 'left':
            return 0
        elif align == 'center':
            return self.width / 2
        elif align == 'right':
            return self.width
        return 0

class Path2D:
    def __init__(self, path=None):
        self.subpaths = []
        self.current_subpath = []
        self.exShape = []
        
        if isinstance(path, Path2D):
            self.subpaths = [sub.copy() for sub in path.subpaths]
            self.current_subpath = self.subpaths[-1].copy() if self.subpaths else []
            self.exShape = path.exShape.copy() if path.exShape else []
        elif isinstance(path, str):
            pass
        else:
            self.beginPath()
    
    def _to_list(self, r):
        if isinstance(r, (int, float)):
            return [r, r]
        return list(r)[:2]

    def _normalize_radii(self, radii, width, height):
        base = [self._to_list(r) for r in radii] if isinstance(radii, (list, tuple)) else [self._to_list(radii)]*4
        base += base*(4//len(base))
        
        if width < 0:
            base[0], base[1] = base[1], base[0]
            base[3], base[2] = base[2], base[3]
        if height < 0:
            base[0], base[3] = base[3], base[0]
            base[1], base[2] = base[2], base[1]
        
        return [
            min(base[0][0], abs(width)/2), 
            min(base[1][0], abs(width)/2),
            min(base[2][0], abs(width)/2), 
            min(base[3][0], abs(width)/2)
        ]

    def beginPath(self):
        self.subpaths = []
        self.current_subpath = []
    
    def moveTo(self, x, y):
        self.current_subpath = [(x, y)]
        self.subpaths.append(self.current_subpath)
    
    def lineTo(self, x, y):
        if not self.current_subpath:
            self.moveTo(x, y)
        else:
            self.current_subpath.append((x, y))

    def closePath(self):
        if self.current_subpath and len(self.current_subpath) >= 1:
            self.current_subpath.append(self.current_subpath[0])
            self.current_subpath = []
    
    def rect(self, x, y, w, h):
        self.subpaths.append([
            (x, y), (x + w, y), (x + w, y + h), (x, y + h), (x, y)
        ])
        self.current_subpath = []
    
    def roundRect(self, x, y, w, h, radii):
        self.exShape.append(
            {
                'type': 'roundRect',
                'x': x,
                'y': y,
                'w': w,
                'h': h,
                'radii': radii,
                'params': {
                    'pos': (x + min(0, w), y + min(0, h)),
                    'size': (abs(w), abs(h)),
                    'radius': self._normalize_radii(radii, w, h)
                }
            }
        )

class ImageData:
    def __init__(self, width, height, data):
        self.width = width
        self.height = height
        self.data = bytes(data)
        self.texture = Texture.create(size=(width, height), colorfmt='rgba')
        self.texture.blit_buffer(self.data, colorfmt='rgba', bufferfmt='ubyte')

class Canvas2DContext(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._textureFlipList = []

        self._lock = False

        self._lastFuncResult = None

        self._methodsPatchList = [
            'fillText', 'strokeText', 'measureText',
            'clearRect', 'fillRect', 'strokeRect', 'putImageData',
            'beginPath', 'closePath', 'moveTo', 'lineTo',
            'rect', 'roundRect', 'fill', 'stroke', 'getImageData',
            'clip', 'rotate', 'scale', 'translate', 'transform',
            'setTransform', 'resetTransform', 'loadTexture',
            'drawImage', 'save', 'restore', 'reset', 'resize'
        ]

        self._propertiesPatchList = [
            'fillStyle',
            'strokeStyle',
            'lineWidth', 
            'font', 
            'textAlign', 
            'textBaseline',
            'filter',
            'globalAlpha',
            'imageSmoothingEnabled'
        ]

        self._needResultMethods = [
            'measureText',
            'loadTexture',
            'getImageData'
        ]

        self._origMethodsList = {}
        self._warppedMethodsList = {}
        self._origPropertiesList = {}
        self._wrappedPropertiesList = {}

        self._combined_matrix = Matrix()

        self._fbo = None

        self.reset()

        self.bind(pos=self._update_rect, size=self._update_rect)

    def __enter__(self):
        self.beginDraw()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.endDraw()
    
    @property
    def lineWidth(self) -> float:
        return self._line_width

    @lineWidth.setter
    def lineWidth(self, value: float) -> None:
        self._line_width = value

    @property
    def textAlign(self) -> str:
        return self._text_align

    @textAlign.setter
    def textAlign(self, value: str) -> None:
        self._text_align = value

    @property
    def textBaseline(self) -> str:
        return self._text_baseline

    @textBaseline.setter
    def textBaseline(self, value: str) -> None:
        self._text_baseline = value

    @property
    def font(self):
        return self._font
    
    @font.setter
    def font(self, value):
        self._font = value
        css_font = CSSFont(value)
        self._font_size = css_font.font_size
        self._font_name = css_font._get_kivy_font_name()

    @property
    def fillStyle(self):
        return self._fill_style
    
    @fillStyle.setter
    def fillStyle(self, color_str):
        if isinstance(color_str, tuple):
            self._fill_style = color_str
        else:
            self._fill_style = CSSColorParser.parse_color(color_str)

    @property
    def strokeStyle(self):
        return self._stroke_style
    
    @strokeStyle.setter
    def strokeStyle(self, color_str):
        if isinstance(color_str, tuple):
            self._stroke_style = color_str
        else:
            self._stroke_style = CSSColorParser.parse_color(color_str)

    @property
    def filter(self) -> str:
        return self._filter

    @filter.setter
    def filter(self, value: str) -> None:
        self._filter = value

    @property
    def globalAlpha(self) -> float:
        return self._globalAlpha

    @globalAlpha.setter
    def globalAlpha(self, value: float) -> None:
        self._globalAlpha = max(0.0, min(1.0, value))

    @property
    def imageSmoothingEnabled(self) -> bool:
        return self._imageSmoothingEnabled

    @imageSmoothingEnabled.setter
    def imageSmoothingEnabled(self, value: bool) -> None:
        self._imageSmoothingEnabled = value

    def _update_rect(self, *args):
        self.___rect.pos = self.pos
        self.___rect.size = self.size

    def _patchAll(self):
        for method in self._methodsPatchList:
            self._wrap_method(method)

        for prop in self._propertiesPatchList:
            self._wrap_property_setter(prop)

    def _restoreAll(self):
        for method in self._methodsPatchList:
            self._restore_method(method)

        for prop in self._propertiesPatchList:
            self._restore_property_setter(prop)

    def _wrap_method(self, method_name):
        if method_name not in self._origMethodsList or method_name not in self._warppedMethodsList:
            self._origMethodsList[method_name] = getattr(self, method_name)
            self._warppedMethodsList[method_name] = self._create_scheduled_method(self._origMethodsList[method_name])
        setattr(self, method_name, self._warppedMethodsList[method_name])

    def _restore_method(self, method_name):
        setattr(self, method_name, self._origMethodsList[method_name])

    def _create_scheduled_method(self, original_func):
        @wraps(original_func)
        def wrapper(*args, **kwargs):
            self._drawInsts.append(
                (original_func, (args, kwargs))
            )
            if original_func.__name__ in self._needResultMethods:
                self.endDraw()
                self.beginDraw()
                
                result = self._lastFuncResult
                self._lastFuncResult = None
                return result
        return wrapper
    
    def _wrap_property_setter(self, prop_name):
        prop = getattr(type(self), prop_name)
        if isinstance(prop, property) and prop.fset:
            if prop_name not in self._origPropertiesList or prop_name not in self._wrappedPropertiesList:
                self._origPropertiesList[prop_name] = prop.fset
                self._wrappedPropertiesList[prop_name] = self._create_scheduled_method(self._origPropertiesList[prop_name])
            new_prop = property(
                fget=prop.fget,
                fset=self._wrappedPropertiesList[prop_name],
                fdel=prop.fdel,
                doc=prop.__doc__
            )
            setattr(type(self), prop_name, new_prop)

    def _restore_property_setter(self, prop_name):
        prop = getattr(type(self), prop_name)
        if isinstance(prop, property) and prop.fset:
            new_prop = property(
                fget=prop.fget,
                fset=self._origPropertiesList[prop_name],
                fdel=prop.fdel,
                doc=prop.__doc__
            )
            setattr(type(self), prop_name, new_prop)

    def _applyMatrix(self):
        Scale(x = 1, y = -1, z = 1)
        Translate(0, -self.height)
        cloneMatrix = Matrix()
        cloneMatrix.set(flat = self._combined_matrix.get())
        MatrixInstruction().matrix = cloneMatrix

    def _exce_draw(self, *args, **kwargs):
        self._restoreAll()
        for inst in self._drawInsts:
            mtd, args = inst
            self._lastFuncResult = mtd(*args[0], **args[1])
        self._lock = False
    
    def _apply_global_alpha(self, color):
        r, g, b, a = color
        return (r, g, b, a * self._globalAlpha)
    
    def _beginClip(self):
        if self._clip_path:
            StencilPush()
            
            Color(1, 1, 1, 1)
            for shape in self._clip_path.exShape:
                shapeType = shape['type']
                if shapeType == 'roundRect':
                    params = shape['params']
                    PushMatrix()

                    Scale(x = 1, y = -1, z = 1, origin=params['pos'])
                    Translate(x = 0, y = -params['size'][1])

                    RoundedRectangle(
                        pos=params['pos'],
                        size=params['size'],
                        radius=params['radius']
                    )

                    PopMatrix()
            
            for subpath in self._clip_path.subpaths:
                if len(subpath) >= 3:
                    vertices = []
                    for point in subpath:
                        vertices.extend([*point, 0, 0])
                    
                    Mesh(
                        vertices=vertices,
                        indices=list(range(len(subpath))),
                        mode='triangle_fan',
                        tex_coords=(0, 0, 1, 0, 1, 1, 0, 1)
                    )
            
            if self._clip_fill_rule == 'evenodd':
                StencilUse(func='equal', ref=1)
            else:
                StencilUse(func='greater', ref=0)

    def _endClip(self):
        if self._clip_path:
            StencilPop()

    def beginDraw(self):
        while self._lock: pass
        self._patchAll()
        self._drawInsts = []

    def endDraw(self):
        self._lock = True
        Clock.schedule_once(self._exce_draw, 0)

    def reset(self):
        self.resetTransform()

        self._fill_style = (0, 0, 0, 1)
        self._stroke_style = (0, 0, 0, 1)
        self._line_width = 1.0
        self._text_align = 'left'
        self._text_baseline = 'alphabetic'
        self._clip_path = None
        self._clip_fill_rule = 'nonzero'
        self._state_stack = []
        self._filter = None
        self._globalAlpha = 1
        self._current_path = Path2D()
        self._font = '10px sans-serif'
        self._font_size = 10
        self._font_name = 'sans-serif'
        self._imageSmoothingEnabled = True
        
        self.canvas.clear()

        with self.canvas:
            Color(1, 1, 1, 1)
            self.___rect = Rectangle(pos=self.pos, size=self.size)
    
    def clearRect(self, x, y, width, height):
        with self.canvas:
            PushMatrix()
            self._applyMatrix()
            Color(1, 1, 1, 1)
            Rectangle(pos=(x, y), size=(width, height))
            PopMatrix()

    def fillRect(self, x, y, width, height):
        with self.canvas:
            PushMatrix()
            self._applyMatrix()

            self._beginClip()
            Color(*self._apply_global_alpha(self.fillStyle))

            Rectangle(
                pos=(x, y),
                size=(width, height),
                tex_coords=(0, 0, 1, 0, 1, 1, 0, 1)
            )

            self._endClip()
            PopMatrix()

    def strokeRect(self, x, y, width, height):
        with self.canvas:
            PushMatrix()
            self._applyMatrix()

            self._beginClip()
            Color(*self._apply_global_alpha(self.strokeStyle))
            
            Line(
                rectangle=(x, y, width, height),
                width=self.lineWidth,
                tex_coords=(0, 0, 1, 0, 1, 1, 0, 1)
            )

            self._endClip()
            PopMatrix()

    def fillText(self, text: str, x: float, y: float, max_width: float = None) -> None:
        label = CoreLabel(
            text=text, 
            font_size=self._font_size, 
            font_name=self._font_name.split(',')[0], 
            valign='top'
        )
        label.refresh()

        texture = label.texture

        text_width, text_height = texture.width, texture.height
        
        scale_factor = 1.0
        if max_width and text_width > max_width:
            scale_factor = max_width / text_width
            text_height *= scale_factor

        if self.textAlign == 'center':
            x -= text_width * scale_factor / 2
        elif self.textAlign == 'right':
            x -= text_width * scale_factor

        metrics = self.measureText(text)
        font_height = metrics.font_height

        match self.textBaseline:
            case 'top':
                y_adjust = 0  # 顶部对齐，无需调整
            case 'hanging':
                y_adjust = -metrics.hangingBaseline
            case 'middle':
                y_adjust = -font_height / 2
            case 'alphabetic':
                y_adjust = -metrics.alphabeticBaseline
            case 'ideographic':
                y_adjust = -metrics.ideographicBaseline
            case 'bottom':
                y_adjust = -font_height
            case _:
                y_adjust = -metrics.alphabeticBaseline
        
        if not self._imageSmoothingEnabled:
            texture.min_filter = 'nearest'
            texture.mag_filter = 'nearest'
            
        with self.canvas:
            PushMatrix()
            self._applyMatrix()

            pos=(x, y + y_adjust)
            size=(text_width * scale_factor, text_height * scale_factor)

            self._beginClip()
            Color(*self._apply_global_alpha(self.fillStyle))

            Rectangle(
                pos=pos,
                size=size,
                texture=texture,
                tex_coords=(0, 0, 1, 0, 1, 1, 0, 1)
            )

            self._endClip()
            PopMatrix()

    def strokeText(self, text: str, x: float, y: float, max_width: float = None) -> None:
        label = CoreLabel(
            text=text,
            font_size=self._font_size,
            font_name=self._font_name.split(',')[0],
            valign='top'
        )
        label.refresh()

        texture = label.texture
        text_width = texture.width
        text_height = texture.height

        scale_factor = 1.0
        if max_width and text_width > max_width:
            scale_factor = max_width / text_width
            text_height *= scale_factor
        
        if self.textAlign == 'center':
            x -= text_width * scale_factor / 2
        elif self.textAlign == 'right':
            x -= text_width * scale_factor

        ascent = self._font_size * 0.8
        descent = self._font_size * 0.2
        total_height = ascent + descent

        match self.textBaseline:
            case 'top':
                y_adjust = 0
            case 'middle':
                y_adjust = -total_height / 2
            case 'bottom':
                y_adjust = -total_height
            case 'alphabetic':
                y_adjust = -ascent
            case _:
                y_adjust = -ascent

        pos = (x, y + y_adjust)
        size = (text_width * scale_factor, text_height * scale_factor)

        radius = int(self.lineWidth)
        offsets = []
        for dx in range(-radius, radius+1):
            for dy in range(-radius, radius+1):
                if dx * dx + dy * dy <= radius * radius:
                    offsets.append((dx, dy))
        
        if not self._imageSmoothingEnabled:
            texture.min_filter = 'nearest'
            texture.mag_filter = 'nearest'

        with self.canvas:
            PushMatrix()
            self._applyMatrix()
            self._beginClip()

            Color(*self._apply_global_alpha(self.strokeStyle))

            for dx, dy in offsets:
                scaled_dx = dx * scale_factor
                scaled_dy = dy * scale_factor
                Rectangle(
                    pos=(pos[0] + scaled_dx, pos[1] + scaled_dy),
                    size=size,
                    texture=texture,
                    tex_coords=(0, 0, 1, 0, 1, 1, 0, 1)
                )
            
            self._endClip()
            PopMatrix()

    def measureText(self, text: str) -> dict:
        label = CoreLabel(
            text=text,
            font_size=self._font_size,
            font_name=self._font_name.split(',')[0],
            valign='top'
        )
        label.refresh()
        return TextMetrics(label, self)

    def beginPath(self) -> None:
        self._current_path = Path2D()

    def closePath(self) -> None:
        self._current_path.closePath()

    def moveTo(self, x: float, y: float) -> None:
        self._current_path.moveTo(x, y)

    def lineTo(self, x, y):
        self._current_path.lineTo(x, y)

    def rect(self, x, y, w, h):
        self._current_path.rect(x, y, w, h)

    def roundRect(self, x: float, y: float, width: float, height: float, radii) -> None:
        self._current_path.roundRect(x, y, width, height, radii)

    def fill(self, path=None, fill_rule: str = None) -> None:
        path = path or self._current_path
        
        with self.canvas:
            PushMatrix()
            self._applyMatrix()

            self._beginClip()
            Color(*self._apply_global_alpha(self.fillStyle))
            
            for shape in self._current_path.exShape:
                shapeType = shape['type']
                if shapeType == 'roundRect':
                    params = shape['params']

                    Scale(x = 1, y = -1, z = 1, origin=params['pos'])
                    Translate(x = 0, y = -params['size'][1])

                    RoundedRectangle(
                        pos=params['pos'],
                        size=params['size'],
                        radius=params['radius']
                    )

            for subpath in path.subpaths:
                if len(subpath) >= 3:
                    vertices = []

                    for point in subpath:
                        x, y = point
                        vertices.extend([x, y, 0, 0])
                    
                    Mesh(
                        vertices=vertices,
                        indices=list(range(len(subpath))),
                        mode='triangle_fan',
                        tex_coords=(0, 1, 1, 1, 1, 0, 0, 0)
                    )
                
            self._endClip()
            PopMatrix()

    def stroke(self) -> None:
        with self.canvas:
            PushMatrix()
            self._applyMatrix()

            self._beginClip()
            Color(*self._apply_global_alpha(self.strokeStyle))

            for shape in self._current_path.exShape:
                shapeType = shape['type']
                if shapeType == 'roundRect':
                    params = shape['params']

                    Scale(x = 1, y = -1, z = 1, origin=params['pos'])
                    Translate(x = 0, y = -params['size'][1])

                    Line(
                        rounded_rectangle=(
                            params['pos'][0], 
                            params['pos'][1], 
                            params['size'][0], 
                            params['size'][1],
                            *params['radius']
                        ),
                        width=self.lineWidth,
                    )
            
            for subpath in self._current_path.subpaths:
                if len(subpath) >= 2:
                    points = []

                    for point in subpath:
                        points.extend(point)

                    Line(
                        points=points, 
                        width=self.lineWidth
                    )
            
            self._endClip()
            PopMatrix()

    def clip(self, path=None, fill_rule: str = 'nonzero') -> None:
        path = path or self._current_path
        self._clip_path = path
        self._clip_fill_rule = fill_rule

    def rotate(self, angle):
        self._combined_matrix.rotate(
            angle = angle,
            x = 0,
            y = 0,
            z = 1
        )

    def scale(self, sx, sy):
        self._combined_matrix.scale(
            x = sx,
            y = sy,
            z = 0
        )

    def translate(self, tx, ty):
        self._combined_matrix.translate(
            x = tx,
            y = ty,
            z = 0
        )

    def transform(self, a, b, c, d, e, f):
        new_matrix = Matrix()
        new_matrix.set(array=[
            [a, b, 0, 0],
            [c, d, 0, 0],
            [0, 0, 1, 0],
            [e, f, 0, 1],
        ])
        self._combined_matrix = self._combined_matrix.multiply(new_matrix)

    def setTransform(self, a, b, c, d, e, f):
        self._combined_matrix.set(array=[
            [a, b, 0, 0],
            [c, d, 0, 0],
            [0, 0, 1, 0],
            [e, f, 0, 1],
        ])
    
    def resetTransform(self):
        self._combined_matrix.identity()

    def loadTexture(self, image):
        if isinstance(image, str):
            texture = CoreImage(image).texture
            return texture
        if isinstance(image, Texture):
            return image
        if hasattr(image, "texture"):
            return image.texture
        raise TypeError("Unsupported image type")
    
    def drawImage(self, image, *args):
        match len(args):
            case 2:
                dx, dy = args
                source_rect = None
                dw, dh = None, None
            case 4:
                dx, dy, dw, dh = args
                source_rect = None
            case 8:
                sx, sy, sw, sh, dx, dy, dw, dh = args
                source_rect = (sx, sy, sw, sh)
            case _:
                raise ValueError("Expected 2, 4 or 8 parameters")

        texture = self.loadTexture(image)

        if source_rect:
            sx, sy, sw, sh = source_rect
            sx = max(0, min(sx, texture.width))
            sy = max(0, min(sy, texture.height))
            sw = max(0, min(sw, texture.width - sx))
            sh = max(0, min(sh, texture.height - sy))
            if sw <= 0 or sh <= 0:
                return
            sy_adj = texture.height - (sy + sh)
            source_region = texture.get_region(sx, sy_adj, sw, sh)
        else:
            sw, sh = texture.size
            source_region = texture
            if len(args) == 2:
                dw, dh = sw, sh
        
        if not self._imageSmoothingEnabled:
            source_region.min_filter = 'nearest'
            source_region.mag_filter = 'nearest'

        if dw is not None and dh is not None and (dw <= 0 or dh <= 0):
            return

        with self.canvas:
            PushMatrix()
            self._applyMatrix()
            self._beginClip()
            Color(1, 1, 1, self.globalAlpha)
            Rectangle(
                texture=source_region,
                pos=(dx, dy),
                size=(dw or sw, dh or sh),
                tex_coords=(0, 0, 1, 0, 1, 1, 0, 1)
            )
            self._endClip()
            PopMatrix()

    def save(self) -> None:
        state = {
            'fill_style': self._fill_style,
            'stroke_style': self._stroke_style,
            'line_width': self._line_width,
            'font': self._font,
            'text_align': self._text_align,
            'text_baseline': self._text_baseline,
            'clip_path': copy.deepcopy(self._clip_path),
            'clip_fill_rule': self._clip_fill_rule,
            'current_path': copy.deepcopy(self._current_path),
            'filter': self._filter,
            'global_alpha': self._globalAlpha,
            'combined_matrix': self._combined_matrix,
            'image_smoothing_enabled': self._imageSmoothingEnabled,
        }
        self._state_stack.append(state)

    def restore(self) -> None:
        state = self._state_stack.pop()
        self._fill_style = state['fill_style']
        self._stroke_style = state['stroke_style']
        self._line_width = state['line_width']
        self._font = state['font']
        self._text_align = state['text_align']
        self._text_baseline = state['text_baseline']
        self._clip_path = state['clip_path']
        self._clip_fill_rule = state['clip_fill_rule']
        self._current_path = copy.deepcopy(state['current_path'])
        self._filter = state['filter']
        self._globalAlpha = state['global_alpha']
        self._combined_matrix = state['combined_matrix']
        self._imageSmoothingEnabled = state['image_smoothing_enabled']
    
    def resize(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.size = (width, height)

    def regFont(self, name: str, path: str) -> None:
        LabelBase.register(
            name=name,
            fn_regular=path
        )
    
    def getImageData(self, x, y, width, height):
        x = max(0, min(x, self.width))
        y = max(0, min(y, self.height))
        width = max(0, min(width, self.width - x))
        height = max(0, min(height, self.height - y))
        
        if width == 0 or height == 0:
            return ImageData(0, 0, b'')

        if self._fbo is None or self._fbo.size != self.size:
            self._fbo = Fbo(size=self.size, clear_color=(0, 0, 0, 0))
        
        with self._fbo:
            ClearColor(1, 1, 1, 1)
            ClearBuffers()
            for instr in self.canvas.children:
                self._fbo.add(instr)
        self._fbo.draw()

        pixels = glReadPixels(
            x, 
            self.height - y - height,
            width, 
            height, 
            GL_RGBA, 
            GL_UNSIGNED_BYTE
        )

        self._fbo.clear()
        
        return ImageData(
            width,
            height, 
            pixels
        )
    
    def putImageData(self, image_data, dx, dy, dirty_x=0, dirty_y=0, dirty_width=None, dirty_height=None):
        img_width = image_data.width
        img_height = image_data.height

        sx = max(0, min(int(dirty_x), img_width))
        sy = max(0, min(int(dirty_y), img_height))
        
        dirty_width = img_width - sx if dirty_width is None else max(0, int(dirty_width))
        sw = max(0, min(dirty_width, img_width - sx))
        
        dirty_height = img_height - sy if dirty_height is None else max(0, int(dirty_height))
        sh = max(0, min(dirty_height, img_height - sy))

        if sw == 0 or sh == 0:
            return

        if not self._imageSmoothingEnabled:
            image_data.texture.min_filter = 'nearest'
            image_data.texture.mag_filter = 'nearest'
        
        with self.canvas:
            PushMatrix()
            self._applyMatrix()
            self._beginClip()
            
            Scale(x = 1, y = -1, z = 1, origin=(dx, dy))
            Translate(x = 0, y = -sh)

            Color(1, 1, 1, self.globalAlpha)
            Rectangle(
                texture=image_data.texture,
                pos=(dx, dy),
                size=(sw, sh),
            )

            self._endClip()
            PopMatrix()
        

if __name__ == '__main__':
    from threading import Thread
    import time, math
    
    ctx = Canvas2DContext()
    class ctxApp(App):
        def build(self, **kwargs): return ctx

    app = ctxApp()
    
    def draw():
        with ctx:
            ico = ctx.loadTexture('Test/icon.ico')
            ctx.regFont('Phigros', 'Test/font.ttf')
        while True:
            with ctx:
                ctx.reset()
                ctx.font = '20px Phigros'
                
                ctx.drawImage(ico, 0, 0, 233, 320)

                imageData = ctx.getImageData(10, 20, 80, 230)
                ctx.putImageData(imageData, 260, 0)
                ctx.putImageData(imageData, 380, 50)
                ctx.putImageData(imageData, 500, 100)
            time.sleep(1 / 60)

    Thread(target = draw, daemon = True).start()
    app.run()