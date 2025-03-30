# coding: utf-8
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.graphics.transformation import Matrix
from kivy.core.window import Window
from kivy.core.text import Label as CoreLabel, LabelBase
from kivy.core.image import Image as CoreImage
from kivy.clock import Clock
from kivy.graphics.fbo import Fbo
from kivy.graphics.texture import Texture, TextureRegion
from kivy.graphics.stencil_instructions import (
    StencilPush, StencilPop, StencilUse, StencilUnUse # REMOVED StencilOp, StencilFunc
)
from kivy.graphics.opengl import (
    glReadPixels, GL_RGBA, GL_UNSIGNED_BYTE
)
from kivy.graphics import (
    RenderContext, RoundedRectangle, Color, Rectangle, Line, ClearColor,
    Mesh, PushMatrix, PopMatrix, ClearBuffers, Callback,
    MatrixInstruction, UpdateNormalMatrix
)
from kivy.utils import get_color_from_hex, platform # 用于潜在的精度问题
from kivy.vector import Vector # 用于向量计算
from kivy.logger import Logger # For better logging

import re
import copy
import math
import warnings

class CSSColorParser:
    COLORS = {
        'black': '#000000', 'silver': '#c0c0c0', 'gray': '#808080', 'white': '#ffffff',
        'maroon': '#800000', 'red': '#ff0000', 'purple': '#800080', 'fuchsia': '#ff00ff',
        'green': '#008000', 'lime': '#00ff00', 'olive': '#808000', 'yellow': '#ffff00',
        'navy': '#000080', 'blue': '#0000ff', 'teal': '#008080', 'aqua': '#00ffff',
        'orange': '#ffa500', 'aliceblue': '#f0f8ff', 'antiquewhite': '#faebd7',
        'aquamarine': '#7fffd4', 'azure': '#f0ffff', 'beige': '#f5f5dc', 'bisque': '#ffe4c4',
        'blanchedalmond': '#ffebcd', 'blueviolet': '#8a2be2', 'brown': '#a52a2a',
        'burlywood': '#deb887', 'cadetblue': '#5f9ea0', 'chartreuse': '#7fff00',
        'chocolate': '#d2691e', 'coral': '#ff7f50', 'cornflowerblue': '#6495ed',
        'cornsilk': '#fff8dc', 'crimson': '#dc143c', 'cyan': '#00ffff', 'darkblue': '#00008b',
        'darkcyan': '#008b8b', 'darkgoldenrod': '#b8860b', 'darkgray': '#a9a9a9',
        'darkgrey': '#a9a9a9', 'darkgreen': '#006400', 'darkkhaki': '#bdb76b',
        'darkmagenta': '#8b008b', 'darkolivegreen': '#556b2f', 'darkorange': '#ff8c00',
        'darkorchid': '#9932cc', 'darkred': '#8b0000', 'darksalmon': '#e9967a',
        'darkseagreen': '#8fbc8f', 'darkslateblue': '#483d8b', 'darkslategray': '#2f4f4f',
        'darkslategrey': '#2f4f4f', 'darkturquoise': '#00ced1', 'darkviolet': '#9400d3',
        'deeppink': '#ff1493', 'deepskyblue': '#00bfff', 'dimgray': '#696969',
        'dimgrey': '#696969', 'dodgerblue': '#1e90ff', 'firebrick': '#b22222',
        'floralwhite': '#fffaf0', 'forestgreen': '#228b22', 'gainsboro': '#dcdcdc',
        'ghostwhite': '#f8f8ff', 'gold': '#ffd700', 'goldenrod': '#daa520',
        'greenyellow': '#adff2f', 'honeydew': '#f0fff0', 'hotpink': '#ff69b4',
        'indianred': '#cd5c5c', 'indigo': '#4b0082', 'ivory': '#fffff0', 'khaki': '#f0e68c',
        'lavender': '#e6e6fa', 'lavenderblush': '#fff0f5', 'lawngreen': '#7cfc00',
        'lemonchiffon': '#fffacd', 'lightblue': '#add8e6', 'lightcoral': '#f08080',
        'lightcyan': '#e0ffff', 'lightgoldenrodyellow': '#fafad2', 'lightgray': '#d3d3d3',
        'lightgrey': '#d3d3d3', 'lightgreen': '#90ee90', 'lightpink': '#ffb6c1',
        'lightsalmon': '#ffa07a', 'lightseagreen': '#20b2aa', 'lightskyblue': '#87cefa',
        'lightslategray': '#778899', 'lightslategrey': '#778899', 'lightsteelblue': '#b0c4de',
        'lightyellow': '#ffffe0', 'limegreen': '#32cd32', 'linen': '#faf0e6',
        'magenta': '#ff00ff', 'mediumaquamarine': '#66cdaa', 'mediumblue': '#0000cd',
        'mediumorchid': '#ba55d3', 'mediumpurple': '#9370db', 'mediumseagreen': '#3cb371',
        'mediumslateblue': '#7b68ee', 'mediumspringgreen': '#00fa9a', 'mediumturquoise': '#48d1cc',
        'mediumvioletred': '#c71585', 'midnightblue': '#191970', 'mintcream': '#f5fffa',
        'mistyrose': '#ffe4e1', 'moccasin': '#ffe4b5', 'navajowhite': '#ffdead',
        'oldlace': '#fdf5e6', 'olivedrab': '#6b8e23', 'orangered': '#ff4500',
        'orchid': '#da70d6', 'palegoldenrod': '#eee8aa', 'palegreen': '#98fb98',
        'paleturquoise': '#afeeee', 'palevioletred': '#db7093', 'papayawhip': '#ffefd5',
        'peachpuff': '#ffdab9', 'peru': '#cd853f', 'pink': '#ffc0cb', 'plum': '#dda0dd',
        'powderblue': '#b0e0e6', 'rebeccapurple': '#663399', 'rosybrown': '#bc8f8f',
        'royalblue': '#4169e1', 'saddlebrown': '#8b4513', 'salmon': '#fa8072',
        'sandybrown': '#f4a460', 'seagreen': '#2e8b57', 'seashell': '#fff5ee',
        'sienna': '#a0522d', 'skyblue': '#87ceeb', 'slateblue': '#6a5acd',
        'slategray': '#708090', 'slategrey': '#708090', 'snow': '#fffafa',
        'springgreen': '#00ff7f', 'steelblue': '#4682b4', 'tan': '#d2b48c',
        'thistle': '#d8bfd8', 'tomato': '#ff6347', 'turquoise': '#40e0d0',
        'violet': '#ee82ee', 'wheat': '#f5deb3', 'whitesmoke': '#f5f5f5',
        'yellowgreen': '#9acd32', 'transparent': '#00000000' # Added transparent
    }
    # More robust regex allowing spaces around numbers/commas/slashes, % signs
    _RGB_RE = re.compile(
        r'^rgba?\(\s*([\d.%]+)\s*[, ]\s*([\d.%]+)\s*[, ]\s*([\d.%]+)\s*'
        r'(?:[,/\s]\s*([\d.%]+)\s*)?\)$', re.IGNORECASE)

    @classmethod
    def parse_color(cls, color_str) -> tuple[float, float, float, float] | None:
        """Parses CSS color string to (r, g, b, a) tuple (0.0-1.0). Returns None on failure."""
        if not isinstance(color_str, str):
             color_str = str(color_str) # Attempt conversion
        color_str = color_str.strip().lower()
        if not color_str:
            return (0.0, 0.0, 0.0, 1.0) # Default to black for empty string

        # Keyword lookup
        if color_str in cls.COLORS:
            try:
                return get_color_from_hex(cls.COLORS[color_str])
            except ValueError:
                 warnings.warn(f"Color '{color_str}' has invalid hex '{cls.COLORS[color_str]}' in mapping.", stacklevel=3)
                 return None # Fallback if hex in mapping is invalid

        # Hex lookup (#rgb, #rgba, #rrggbb, #rrggbbaa)
        if color_str.startswith('#'):
            try:
                return get_color_from_hex(color_str)
            except ValueError:
                 warnings.warn(f"Invalid hex color format: '{color_str}'.", stacklevel=3)
                 return None # Fallback for invalid hex

        # RGB/RGBA lookup
        if color_str.startswith(('rgb(', 'rgba(')):
            return cls._parse_rgb(color_str)

        # Future: HSL/HSLA?

        # Unrecognized format
        warnings.warn(f"Unrecognized color format: '{color_str}'.", stacklevel=2)
        return None

    @classmethod
    def _parse_rgb(cls, color_str) -> tuple[float, float, float, float] | None:
        match = cls._RGB_RE.match(color_str)
        if not match:
            warnings.warn(f"Invalid RGB(A) format: '{color_str}'.", stacklevel=3)
            return None # Fallback

        try:
            r_str, g_str, b_str, a_str = match.groups()

            r = cls._parse_component(r_str, 255.0)
            g = cls._parse_component(g_str, 255.0)
            b = cls._parse_component(b_str, 255.0)
            # Alpha defaults to 1.0 if not provided
            a = cls._parse_component(a_str, 1.0) if a_str is not None else 1.0

            # Check if any component failed parsing (returned None)
            if r is None or g is None or b is None or a is None:
                # Warning issued inside _parse_component
                raise ValueError("Component parsing failed")

            return (r, g, b, a)
        except Exception as e:
            warnings.warn(f"Error parsing RGB(A) components in '{color_str}': {e}.", stacklevel=3)
            return None # Fallback

    @classmethod
    def _parse_component(cls, component_str, max_val) -> float | None:
        """Parses a single R,G,B or A component string. Returns float 0-1 or None."""
        if component_str is None: return None # Should not happen with regex, but check
        component_str = component_str.strip().lower()
        try:
            val = 0.0
            is_percent = component_str.endswith('%')
            num_str = component_str[:-1] if is_percent else component_str
            num = float(num_str) # Can raise ValueError

            # Check for non-finite numbers early
            if not math.isfinite(num):
                 warnings.warn(f"Non-finite number in color component: '{component_str}'.", stacklevel=4)
                 return None

            if is_percent:
                # Percentage values (relative to 1.0)
                val = num / 100.0
            else:
                # Numeric value (0-255 for RGB, 0-1 for Alpha)
                val = num / max_val

            # Clamp value between 0 and 1
            return max(0.0, min(1.0, val))
        except (ValueError, TypeError):
             warnings.warn(f"Invalid color component value: '{component_str}'.", stacklevel=4)
             return None

class CSSFont:
    _FONT_RE = re.compile(
        r"^\s*"
        r"(?:(normal|italic|oblique(?: \s*(-?\d+(?:\.\d+)?(?:deg|grad|rad|turn)))?)\s+)?"  # Style
        r"(?:(normal|small-caps)\s+)?"  # Variant
        r"(?:(normal|bold|lighter|bolder|[1-9]00)\s+)?"  # Weight
        r"(?:(normal|ultra-condensed|extra-condensed|condensed|semi-condensed|semi-expanded|expanded|extra-expanded|ultra-expanded)\s+)?" # Stretch
        r"(\d+(?:\.\d*)?)(px|pt|pc|in|cm|mm|em|rem|%|vw|vh|vmin|vmax)" # Size and Unit
        r"(?:\s*\/\s*(normal|(?:\d+(?:\.\d*)?)(?:px|pt|pc|in|cm|mm|em|rem|%|vw|vh|vmin|vmax)?)\s+)?" # Line height
        r"(.+)"  # Font Family
        r"\s*$",
        re.IGNORECASE
    )
    _UNIT_CONVERSIONS_PX = {
        'px': 1.0, 'pt': 96.0 / 72.0, 'pc': (96.0 / 72.0) * 12.0, 'in': 96.0,
        'cm': 96.0 / 2.54, 'mm': 96.0 / 25.4, 'em': 16.0, 'rem': 16.0, '%': 0.16,
        # Approximate viewport units based on default font size (16px)
        'vw': Window.width * 0.01 if Window else 16.0 * 0.01, # Use Window width if available
        'vh': Window.height * 0.01 if Window else 16.0 * 0.01, # Use Window height if available
        # vmin/vmax are harder without context, approximate using %
        'vmin': min(Window.width, Window.height) * 0.01 if Window else 16.0 * 0.01,
        'vmax': max(Window.width, Window.height) * 0.01 if Window else 16.0 * 0.01,
    }
    _DEFAULT_FONT_SIZE_PX = 10.0
    _DEFAULT_FONT_FAMILY = ['sans-serif'] # Kivy uses 'Roboto' by default, maybe match?

    def __init__(self, font_str='10px sans-serif'):
        # Set initial defaults before parsing
        self.font_str = font_str # Store the original attempt
        self.font_style = 'normal'
        self.font_variant = 'normal'
        self.font_weight = 'normal'
        self.font_stretch = 'normal'
        self.font_size_px = self._DEFAULT_FONT_SIZE_PX
        self.line_height_str = 'normal'
        self.font_family = self._DEFAULT_FONT_FAMILY[:]

        # Attempt to parse the provided string
        if not self.parse_font_str(font_str):
             # If initial parse fails, reset explicitly to safe defaults
             self.font_str = f"{self._DEFAULT_FONT_SIZE_PX}px {self._DEFAULT_FONT_FAMILY[0]}"
             self.font_style = 'normal'
             self.font_variant = 'normal'
             self.font_weight = 'normal'
             self.font_stretch = 'normal'
             self.font_size_px = self._DEFAULT_FONT_SIZE_PX
             self.line_height_str = 'normal'
             self.font_family = self._DEFAULT_FONT_FAMILY[:]

    def parse_font_str(self, font_str) -> bool:
        """Parses CSS font string. Updates instance attributes on success. Returns True/False."""
        if not isinstance(font_str, str): font_str = str(font_str)

        match = self._FONT_RE.match(font_str)
        simple_match = None
        size_val_str, size_unit, family_str = None, None, None

        if not match:
            # Try simpler regex for just size + family (common case)
            simple_match = re.match(r"^\s*(\d+(?:\.\d*)?)(px|pt|pc|in|cm|mm|em|rem|%|vw|vh|vmin|vmax)\s+(.+)\s*$", font_str, re.IGNORECASE)
            if simple_match:
                 size_val_str, size_unit, family_str = simple_match.groups()
                 # If using simple match, reset other properties to their defaults
                 new_style = 'normal'; new_variant = 'normal'; new_weight = 'normal'; new_stretch = 'normal'; new_lh = 'normal'
            else:
                 # Neither regex matched, parsing failed
                 warnings.warn(f"Could not parse font string: '{font_str}'. Using previous/default font settings.", stacklevel=2)
                 return False # Indicate parsing failure
        else:
             # Full regex matched
             style_group, _oblique_angle, variant, weight, stretch, size_val_str, size_unit, \
             line_height_group, family_str = match.groups()

             # Set defaults first, then override with matched groups (handles missing optional parts)
             new_style = 'normal'; new_variant = 'normal'; new_weight = 'normal'; new_stretch = 'normal'; new_lh = 'normal'
             if style_group: new_style = style_group.lower().split()[0] # Handle 'italic 15deg' -> 'italic'
             if variant: new_variant = variant.lower()
             if weight: new_weight = weight.lower()
             if stretch: new_stretch = stretch.lower()
             if line_height_group: new_lh = line_height_group.lower()

        # --- Parse Size (common to both successful regex paths) ---
        new_size_px = self._DEFAULT_FONT_SIZE_PX # Default if parsing fails
        try:
            size_val = float(size_val_str)
            if not math.isfinite(size_val): raise ValueError("Non-finite size value")
            unit = size_unit.lower()
            conversion = self._UNIT_CONVERSIONS_PX.get(unit)

            if conversion is None:
                 warnings.warn(f"Unsupported font size unit: '{unit}' in '{font_str}'. Using default size.", stacklevel=3)
            else:
                # Base size for relative units (em, rem, %) - use default size? Or inherited? Use default for now.
                base_size = self._DEFAULT_FONT_SIZE_PX
                if unit == '%': new_size_px = base_size * (size_val / 100.0)
                elif unit in ('em', 'rem'): new_size_px = base_size * size_val
                elif unit in ('vw', 'vh', 'vmin', 'vmax'):
                     # Update viewport units based on current Window size if possible
                     current_vw = Window.width * 0.01 if Window else base_size * 0.01
                     current_vh = Window.height * 0.01 if Window else base_size * 0.01
                     if unit == 'vw': new_size_px = current_vw * size_val
                     elif unit == 'vh': new_size_px = current_vh * size_val
                     elif unit == 'vmin': new_size_px = min(current_vw, current_vh) * size_val
                     elif unit == 'vmax': new_size_px = max(current_vw, current_vh) * size_val
                     warnings.warn(f"Viewport units ({unit}) for font size depend on window size and may change.", stacklevel=3)
                else: # Absolute units (px, pt, pc, in, cm, mm)
                    new_size_px = size_val * conversion

            # Ensure font size is positive
            new_size_px = max(1.0, new_size_px)
        except (ValueError, TypeError) as e:
             warnings.warn(f"Invalid font size value ('{size_val_str}' or unit '{size_unit}') in '{font_str}': {e}. Using default size.", stacklevel=3)
             new_size_px = self._DEFAULT_FONT_SIZE_PX # Reset on error

        # --- Parse Family (common to both successful regex paths) ---
        new_families = self._DEFAULT_FONT_FAMILY[:] # Default if parsing fails
        families_raw = [f.strip().strip('\'"') for f in family_str.split(',') if f.strip()]
        if families_raw:
            new_families = families_raw
        else:
             warnings.warn(f"No valid font families found in '{family_str}' part of '{font_str}'. Using default.", stacklevel=3)

        # --- Update instance attributes only if parsing succeeded ---
        self.font_style = new_style
        self.font_variant = new_variant
        self.font_weight = new_weight
        self.font_stretch = new_stretch
        self.line_height_str = new_lh
        self.font_size_px = new_size_px
        self.font_family = new_families
        self.font_str = font_str # Store original successful string

        return True # Indicate parsing success

    @property
    def is_italic(self) -> bool: return self.font_style in ('italic', 'oblique')
    @property
    def is_bold(self) -> bool:
        weight = self.font_weight
        if weight in ('bold', 'bolder'): return True
        try: return int(weight) >= 700 # Numeric weights 700+ are bold
        except (ValueError, TypeError): return False

    @property
    def kivy_font_name(self) -> str:
        """Finds the first registered font family or falls back."""
        registered_fonts = [] # Default to empty list if check fails
        try:
            # Attempt to get registered fonts
            registered_fonts = LabelBase.get_registered_fonts()
        except AttributeError: # Catch if LabelBase or method not ready
            # Log a DEBUG message instead of warning, as this might happen normally during init
            Logger.debug("CSSFont: LabelBase not fully initialized during font check, assuming no fonts registered yet.")
            # Keep registered_fonts as []
        except Exception as e: # Catch other unexpected errors
            Logger.warning(f"CSSFont: Unexpected error getting registered fonts: {e}")
            # Keep registered_fonts as []

        # Find first registered font family from the list
        for family in self.font_family:
            if family in registered_fonts:
                return family

        # Fallback: Return the first family name specified
        if not self.font_family:
            return self._DEFAULT_FONT_FAMILY[0]

        first_family = self.font_family[0]
        # Optional: Reduce warning noise, only warn if explicitly needed later
        # if first_family not in registered_fonts:
        #    Logger.debug(f"Canvas: Font family '{first_family}' (from '{self.font_str}') not registered with Kivy. Fallback behavior may apply.")
        return first_family

class TextMetrics:
    """Represents the measurement metrics for a piece of text."""
    def __init__(self, text: str, ctx: 'Canvas2DContext'):
        # Store context state at time of measurement
        self._measured_text = text
        self._measured_font_size = ctx._font_size
        self._measured_font_name = ctx._font_name # Derived Kivy font name from CSSFont
        self._measured_bold = ctx._font_bold
        self._measured_italic = ctx._font_italic

        # Create a Kivy CoreLabel instance for measurement
        # Note: Kivy's font resolution might differ slightly from browser CSS logic.
        self._label = CoreLabel(
            text=text,
            font_size=self._measured_font_size,
            font_name=self._measured_font_name, # Use the name CSSFont determined
            bold=self._measured_bold,
            italic=self._measured_italic,
        )
        self._label.refresh() # Render the label internally to get size/texture

        # Get width using Kivy's extent calculation
        self._text_width = self._label.get_extents(text)[0]
        # Get height from the rendered texture size (approximates bounding box)
        self._text_height = self._label.texture.height if self._label.texture else self._measured_font_size

        # --- Approximate Baseline Metrics ---
        # Kivy doesn't expose detailed font metrics like ascent/descent easily.
        # We approximate these based on font size and bounding box height.
        # These are HEURISTICS and will vary significantly between fonts.
        self._font_size_px = self._measured_font_size
        # Approx Asc/Desc based on measured height (may include padding)
        self._actual_ascent = self._text_height * 0.8  # Guess: ~80% of bounding box above baseline
        self._actual_descent = self._text_height * 0.2 # Guess: ~20% below baseline
        # Approx Asc/Desc based on font size (conceptual metrics)
        self._font_ascent = self._font_size_px * 0.8 # Guess: ~80% of font size above baseline
        self._font_descent = self._font_size_px * 0.3 # Guess: ~30% below baseline (allows deeper descenders)
        self._extents = self._label.get_extents(self._label.text) if self._label.text else None
        self._texture = self._label.texture if self._label else None

    @property
    def width(self) -> float:
        """Measured width of the text."""
        return self._text_width
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
    # --- Bounding Box Metrics (Approximated) ---
    @property
    def actualBoundingBoxAscent(self) -> float:
        """Approx distance from alphabetic baseline to highest point of text."""
        return self._actual_ascent
    @property
    def actualBoundingBoxDescent(self) -> float:
        """Approx distance from alphabetic baseline to lowest point of text."""
        return self._actual_descent
    @property
    def actualBoundingBoxLeft(self) -> float:
        """Approx distance from alignment point (x) to left edge of text bounding box.
           Assumes textAlign='left'/'start' for basic metrics."""
        # TODO: Could adjust based on textAlign, but 'width' is the primary metric.
        return 0.0
    @property
    def actualBoundingBoxRight(self) -> float:
        """Approx distance from alignment point (x) to right edge of text bounding box.
           Assumes textAlign='left'/'start'."""
        return self.width

    # --- Font Metrics (Approximated based on font size) ---
    @property
    def fontBoundingBoxAscent(self) -> float:
        """Approx distance from baseline to top of font's conceptual bounding box."""
        return self._font_ascent
    @property
    def fontBoundingBoxDescent(self) -> float:
        """Approx distance from baseline to bottom of font's conceptual bounding box."""
        return self._font_descent

    # --- Common Aliases (using actual bounding box approximations) ---
    @property
    def ascent(self) -> float: return self.actualBoundingBoxAscent
    @property
    def descent(self) -> float: return self.actualBoundingBoxDescent

class Path2D:
    def __init__(self, path=None):
        # subpaths stores lists of points [ (x,y), ... ] or dicts for shapes
        self.subpaths: list[list[tuple[float, float]] | dict] = []
        # State for the current point list being built
        self._current_point_subpath: list[tuple[float, float]] | None = None
        self._current_point_subpath_start: tuple[float, float] | None = None

        if isinstance(path, Path2D):
            # Deep copy path data if constructed from another Path2D
            self.subpaths = copy.deepcopy(path.subpaths)
            # Try to set the current subpath state based on the copy
            last_point_path_info = self._find_last_point_subpath()
            if last_point_path_info:
                 index, _ = last_point_path_info
                 self._current_point_subpath = self.subpaths[index] # Reference the copied list
                 self._current_point_subpath_start = self._current_point_subpath[0] if self._current_point_subpath else None
            else: self._reset_current_point_subpath() # No point lists found in copy
        elif isinstance(path, str):
            # TODO: Implement SVG path data string parsing (complex)
            warnings.warn("Path2D constructor does not support SVG path data strings.", stacklevel=2)
            self._reset_current_point_subpath()
        else:
            # Default constructor or invalid path argument
            self._reset_current_point_subpath()

    def _find_last_point_subpath(self) -> tuple[int, list[tuple[float, float]]] | None:
        """Finds the index and reference of the last subpath that is a list of points."""
        for i in range(len(self.subpaths) - 1, -1, -1):
            subpath = self.subpaths[i]
            if isinstance(subpath, list): # Check if it's a list (representing points)
                return i, subpath
        return None

    def _reset_current_point_subpath(self):
        """Resets the state tracking the current list of points being built."""
        self._current_point_subpath = None
        self._current_point_subpath_start = None

    def _ensure_current_point_subpath(self) -> list[tuple[float, float]]:
        """Gets the current list subpath, creating a new one if necessary."""
        if self._current_point_subpath is None:
            self._current_point_subpath = []
            self.subpaths.append(self._current_point_subpath)
            self._current_point_subpath_start = None # Reset start point for the new subpath list
        return self._current_point_subpath

    def _add_point(self, x: float, y: float):
        """Adds a point to the current list subpath, validating coordinates."""
        try:
            px, py = float(x), float(y)
            # Ensure coordinates are finite numbers
            if not (math.isfinite(px) and math.isfinite(py)):
                 warnings.warn(f"Ignoring non-finite point ({x}, {y}) in path command.", stacklevel=3)
                 return # Skip adding non-finite points

            current_list = self._ensure_current_point_subpath()
            point = (px, py)
            if not current_list: # First point in this new subpath list
                self._current_point_subpath_start = point
            current_list.append(point)
        except (ValueError, TypeError):
             # Catch potential errors from float()
             warnings.warn(f"Invalid coordinate type for point ({x}, {y}). Ignoring point.", stacklevel=3)

    def closePath(self):
        """Closes the current list subpath by connecting the last point to the start."""
        if (self._current_point_subpath and self._current_point_subpath_start and
            len(self._current_point_subpath) > 1): # Need at least two points to close
            current_point = self._current_point_subpath[-1]
            start_point = self._current_point_subpath_start
            # Check if already closed (or very close) to avoid duplicate points
            # Use a small epsilon for float comparison
            if math.dist(current_point, start_point) > 1e-9:
                 self._current_point_subpath.append(start_point) # Add the start point to close

        # Closing a path finishes the current point list subpath.
        # Subsequent commands like lineTo should effectively start a new subpath
        # (often implicitly via a moveTo needed by the drawing logic).
        self._reset_current_point_subpath()

    def moveTo(self, x: float, y: float):
        """Starts a new list subpath at (x, y)."""
        # End previous list subpath implicitly if one was active
        self._reset_current_point_subpath()
        # Start the new one by adding the first point
        self._add_point(x, y)

    def lineTo(self, x: float, y: float):
        """Adds a line segment to (x, y). Starts a subpath implicitly if none exists."""
        if self._current_point_subpath is None or not self._current_point_subpath:
            # Spec: If there's no subpath or current point, lineTo should implicitly moveTo(x, y).
            # This creates a zero-length subpath initially, which is fine.
            warnings.warn("lineTo called without current point; implicitly starting new subpath at target.", stacklevel=3)
            self.moveTo(x, y)
            # After the implicit moveTo, we *don't* add the point again here.
        else:
            # Add the point to the existing list subpath
            self._add_point(x, y)

    # --- Curve Approximations (Placeholders - Need Subdivision for Accurate Rendering) ---
    # TODO: Implement curve subdivision (e.g., using De Casteljau's algorithm or similar)
    #       for bezierCurveTo, quadraticCurveTo, arc, arcTo, ellipse.

    def bezierCurveTo(self, cp1x, cp1y, cp2x, cp2y, x, y):
        """Adds cubic Bezier curve. Placeholder: Adds line segment to end point."""
        # Check for current point before adding endpoint
        if self._current_point_subpath is None or not self._current_point_subpath:
             # Spec implies this is equivalent to moveTo(cp1x, cp1y) then curve?
             # Simpler: move to the final point if no start point exists.
             warnings.warn("bezierCurveTo called without current point; implicitly moving to end point.", stacklevel=3)
             self.moveTo(x, y)
        else:
            self._add_point(x, y) # Add end point as approximation
        warnings.warn("Path2D.bezierCurveTo() uses line segment approximation. Curves will not be smooth.", stacklevel=2)

    def quadraticCurveTo(self, cpx, cpy, x, y):
        """Adds quadratic Bezier curve. Placeholder: Adds line segment to end point."""
        if self._current_point_subpath is None or not self._current_point_subpath:
             warnings.warn("quadraticCurveTo called without current point; implicitly moving to end point.", stacklevel=3)
             self.moveTo(x, y)
        else:
            self._add_point(x, y)
        warnings.warn("Path2D.quadraticCurveTo() uses line segment approximation. Curves will not be smooth.", stacklevel=2)

    def arc(self, x, y, radius, startAngle, endAngle, counterclockwise=False):
        """Adds circular arc. Placeholder: Lines to start/end points."""
        try:
             # Validation moved here
             xf, yf, rf, saf, eaf = map(float, [x, y, radius, startAngle, endAngle])
             if not all(math.isfinite(v) for v in [xf, yf, rf, saf, eaf]): raise TypeError("Non-finite arguments")
             if rf < 0: raise ValueError("Radius cannot be negative") # Simulates IndexSizeError

             # Calculate start and end points of the arc segment
             start_x = xf + rf * math.cos(saf)
             start_y = yf + rf * math.sin(saf)
             end_x = xf + rf * math.cos(eaf)
             end_y = yf + rf * math.sin(eaf)

             if self._current_point_subpath is None or not self._current_point_subpath:
                 # Spec: If path empty, add start point, then arc.
                 self.moveTo(start_x, start_y)
             else:
                 # Spec: If path not empty, add straight line segment from current point to arc's start point.
                 last_pt = self._current_point_subpath[-1]
                 # Only add line if start point is different from current point
                 if math.dist(last_pt, (start_x, start_y)) > 1e-9:
                     self.lineTo(start_x, start_y)

             # Add the arc segment (approximated by a line to the end point)
             self.lineTo(end_x, end_y) # TODO: Replace with actual arc point generation
             warnings.warn("Path2D.arc() uses line segment approximation. Arcs will not be smooth.", stacklevel=2)
        except (ValueError, TypeError) as e: raise e # Propagate validation errors

    def arcTo(self, x1, y1, x2, y2, radius):
        """Adds arc segment using tangent lines. Placeholder: Lines to control points."""
        try:
             # Validation
             x1f, y1f, x2f, y2f, rf = map(float, [x1, y1, x2, y2, radius])
             if not all(math.isfinite(v) for v in [x1f, y1f, x2f, y2f, rf]): raise TypeError("Non-finite arguments")
             if rf < 0: raise ValueError("Radius cannot be negative")

             if self._current_point_subpath is None or not self._current_point_subpath:
                  # Spec: If path is empty, it's equivalent to moveTo(x1, y1).
                  warnings.warn("arcTo called without current point; implicitly moving to first control point (x1, y1).", stacklevel=3)
                  self.moveTo(x1f, y1f)
             else:
                  # Get current point P0
                  p0 = self._current_point_subpath[-1]
                  # TODO: Implement actual arcTo logic:
                  # 1. Find tangent lines P0->P1 and P1->P2.
                  # 2. Calculate tangent points T0 and T1 on these lines.
                  # 3. Add line from P0 to T0.
                  # 4. Add arc segment from T0 to T1.
                  # Placeholder: Add lines to control points as rough approximation
                  if math.dist(p0, (x1f, y1f)) > 1e-9:
                      self.lineTo(x1f, y1f) # Line to first control point
                  self.lineTo(x2f, y2f) # Line to second control point
                  warnings.warn("Path2D.arcTo() uses line segment approximation. Behavior is incorrect.", stacklevel=2)
        except (ValueError, TypeError) as e: raise e

    def ellipse(self, x, y, radiusX, radiusY, rotation, startAngle, endAngle, counterclockwise=False):
        """Adds elliptical arc. Placeholder: Lines to rotated start/end points."""
        try:
            # Validation
            xf, yf, rxf, ryf, rotf, saf, eaf = map(float, [x, y, radiusX, radiusY, rotation, startAngle, endAngle])
            if not all(math.isfinite(v) for v in [xf, yf, rxf, ryf, rotf, saf, eaf]): raise TypeError("Non-finite arguments")
            if rxf < 0 or ryf < 0: raise ValueError("Radii cannot be negative")

            # Placeholder: Calculate rotated start and end points
            cos_rot = math.cos(rotf); sin_rot = math.sin(rotf)
            # Calculate unrotated points relative to center
            start_x_unrot = rxf * math.cos(saf); start_y_unrot = ryf * math.sin(saf)
            end_x_unrot = rxf * math.cos(eaf); end_y_unrot = ryf * math.sin(eaf)
            # Apply rotation and translate to center
            start_x = xf + start_x_unrot * cos_rot - start_y_unrot * sin_rot
            start_y = yf + start_x_unrot * sin_rot + start_y_unrot * cos_rot
            end_x = xf + end_x_unrot * cos_rot - end_y_unrot * sin_rot
            end_y = yf + end_x_unrot * sin_rot + end_y_unrot * cos_rot

            # Add points similar to arc() logic
            if self._current_point_subpath is None or not self._current_point_subpath:
                self.moveTo(start_x, start_y)
            else:
                 last_pt = self._current_point_subpath[-1]
                 if math.dist(last_pt, (start_x, start_y)) > 1e-9:
                     self.lineTo(start_x, start_y)
            self.lineTo(end_x, end_y) # TODO: Replace with actual ellipse point generation
            warnings.warn("Path2D.ellipse() uses line segment approximation. Ellipses will not be smooth.", stacklevel=2)
        except (ValueError, TypeError) as e: raise e

    def rect(self, x: float, y: float, w: float, h: float):
        """Adds a rectangle as a new closed list subpath."""
        try:
            xf, yf, wf, hf = map(float, [x, y, w, h])
            # Spec allows non-finite coords but rect has no effect? Let's check here.
            if not all(math.isfinite(v) for v in [xf, yf, wf, hf]):
                 warnings.warn(f"Ignoring rect() with non-finite values: {(x,y,w,h)}.", stacklevel=3)
                 return

            # rect creates a new subpath implicitly, starting with moveTo
            self.moveTo(xf, yf)
            self.lineTo(xf + wf, yf)
            self.lineTo(xf + wf, yf + hf)
            self.lineTo(xf, yf + hf)
            self.closePath() # Closes this specific rectangle subpath list
        except (ValueError, TypeError):
             warnings.warn(f"Invalid rect values: {(x,y,w,h)}. Ignoring call.", stacklevel=2)
             # Don't raise, just ignore as per spec for most path errors

    def _normalize_radius(self, r) -> list[float]:
        """Helper for roundRect: normalizes a single radius or pair/list element."""
        if isinstance(r, (int, float)):
            val = max(0.0, float(r)) # Ensure radius is non-negative
            return [val, val] # [Horizontal, Vertical]
        elif isinstance(r, dict) and 'x' in r and 'y' in r: # Handle {x: h, y: v} format
             try:
                 horz = max(0.0, float(r['x']))
                 vert = max(0.0, float(r['y']))
                 return [horz, vert]
             except (ValueError, TypeError, KeyError):
                 warnings.warn(f"Invalid radius dict element: {r}. Using [0,0].", stacklevel=4)
                 return [0.0, 0.0]
        elif isinstance(r, (list, tuple)) and len(r) >= 1:
            try:
                horz = max(0.0, float(r[0]))
                # Default vertical = horizontal if only one value provided
                vert = max(0.0, float(r[1])) if len(r) > 1 else horz
                return [horz, vert]
            except (ValueError, TypeError, IndexError):
                warnings.warn(f"Invalid radius element in list/tuple: {r}. Using [0,0].", stacklevel=4)
                return [0.0, 0.0]
        else:
            warnings.warn(f"Invalid radius type: {type(r)}. Using [0,0].", stacklevel=4)
            return [0.0, 0.0]

    def roundRect(self, x: float, y: float, w: float, h: float, radii: float | dict | list | tuple = 0) -> None:
        """Adds a rounded rectangle shape definition to the path's subpath list."""
        try:
            xf, yf, wf, hf = map(float, [x, y, w, h])
            # Spec requires finite coords/size
            if not all(math.isfinite(v) for v in [xf, yf, wf, hf]):
                 warnings.warn(f"Ignoring roundRect() with non-finite position/size.", stacklevel=2)
                 return
        except (ValueError, TypeError):
            warnings.warn(f"Invalid position/size arguments for roundRect.", stacklevel=2)
            return

        r = radii
        r_normalized: list[list[float]] = [] # List of [horz, vert] for TL, TR, BR, BL corners

        # --- Normalize radii input according to spec ---
        if isinstance(r, (int, float)): # Single number -> all corners same H/V radius
             r_normalized = [self._normalize_radius(r)] * 4
        elif isinstance(r, (list, tuple)):
            num_radii = len(r)
            if num_radii == 0: r_normalized = [[0.0, 0.0]] * 4 # No radii provided
            elif num_radii == 1: r_normalized = [self._normalize_radius(r[0])] * 4 # One element for all corners
            elif num_radii == 2: r0, r1 = self._normalize_radius(r[0]), self._normalize_radius(r[1]); r_normalized = [r0, r1, r0, r1] # TL/BR, TR/BL
            elif num_radii == 3: r0, r1, r2 = self._normalize_radius(r[0]), self._normalize_radius(r[1]), self._normalize_radius(r[2]); r_normalized = [r0, r1, r2, r1] # TL, TR/BL, BR
            elif num_radii >= 4: r_normalized = [self._normalize_radius(r[i]) for i in range(4)] # TL, TR, BR, BL
            # else: Should not happen with list/tuple check, handled by _normalize_radius errors
        elif isinstance(r, dict): # Allow single {x:h, y:v} for all corners? Spec unclear, let's support it.
             r_norm = self._normalize_radius(r)
             r_normalized = [r_norm] * 4
             warnings.warn("Using single dict for roundRect radii applies it to all corners.", stacklevel=3)
        else:
            warnings.warn(f"Invalid radii type: {type(r)}. Using 0 radius.", stacklevel=2)
            r_normalized = [[0.0, 0.0]] * 4

        # Extract individual corner radii [Horizontal, Vertical]
        [[tl_hr, tl_vr], [tr_hr, tr_vr], [br_hr, br_vr], [bl_hr, bl_vr]] = r_normalized

        # Handle negative width/height by adjusting x/y and using absolute values for calculations
        draw_x = xf if wf >= 0 else xf + wf
        draw_y = yf if hf >= 0 else yf + hf
        abs_w, abs_h = abs(wf), abs(hf)

        # --- Scale radii down if they overlap (as per CSS spec) ---
        epsilon = 1e-9 # Small value for float comparisons
        if abs_w < epsilon or abs_h < epsilon:
            # If width or height is zero, all radii effectively become zero
            tl_hr, tl_vr = tr_hr, tr_vr = br_hr, br_vr = bl_hr, bl_vr = 0.0, 0.0
        else:
            # Calculate scale factors based on adjacent horizontal/vertical radii sums vs width/height
            scale_top = abs_w / (tl_hr + tr_hr) if (tl_hr + tr_hr) > abs_w else 1.0
            scale_right = abs_h / (tr_vr + br_vr) if (tr_vr + br_vr) > abs_h else 1.0
            scale_bot = abs_w / (bl_hr + br_hr) if (bl_hr + br_hr) > abs_w else 1.0
            scale_left = abs_h / (tl_vr + bl_vr) if (tl_vr + bl_vr) > abs_h else 1.0
            # Find the minimum scale factor needed across all sides and corners
            min_scale = min(scale_top, scale_right, scale_bot, scale_left, 1.0) # Include 1.0 ensure scale doesn't exceed 1

            # Apply the scale factor if it's less than 1 (i.e., overlap occurred)
            if min_scale < 1.0:
                tl_hr *= min_scale; tl_vr *= min_scale
                tr_hr *= min_scale; tr_vr *= min_scale
                br_hr *= min_scale; br_vr *= min_scale
                bl_hr *= min_scale; bl_vr *= min_scale

        # Store the roundRect data as a dictionary in subpaths
        # This separates it from point-list subpaths.
        # Kivy's RoundedRectangle uses a single radius list [tl, tr, br, bl].
        # We store detailed radii for potential future use (e.g., complex shaders)
        # but also provide the simplified list for current Kivy rendering.
        shape_data = {
            'type': 'roundRect',
            'x': draw_x, 'y': draw_y, 'w': abs_w, 'h': abs_h,
            # Store detailed pairs for potential future use
            'radii_detail': [[tl_hr, tl_vr], [tr_hr, tr_vr], [br_hr, br_vr], [bl_hr, bl_vr]],
            # Kivy simplified radii: Use horizontal radii for TL, TR, BR, BL corners by convention?
            # Or average? Let's use horizontal radii for now. Kivy's own spec is slightly ambiguous.
            # Kivy examples suggest radius=[10, 20, 30, 40] maps to TL, TR, BR, BL.
            'kivy_radii': [tl_hr, tr_hr, br_hr, bl_hr]
        }
        self.subpaths.append(shape_data)

        # roundRect defines a complete shape and conceptually starts a new path (like rect).
        # Reset the current point list subpath state.
        self._reset_current_point_subpath()

    def addPath(self, path: 'Path2D', transform: Matrix | None = None) -> None:
        """Adds another Path2D object's subpaths to this one, optionally transformed."""
        if not isinstance(path, Path2D):
            # Spec requires TypeError
            raise TypeError("Argument 1 of Path2D.addPath is not an object.")
        if not path.subpaths:
            return # Nothing to add

        # Make a deep copy of the subpaths from the source path to avoid modifying the original
        # and allow independent transformation.
        copied_subpaths = copy.deepcopy(path.subpaths)

        # Apply transformation if provided
        if transform:
            if not isinstance(transform, Matrix):
                 # Should ideally check for DOMMatrix interface, but use Kivy Matrix here
                 warnings.warn("Invalid transform provided to addPath; must be a Kivy Matrix.", stacklevel=3)
                 transform = None # Ignore invalid transform

        if transform:
            # Apply the transformation matrix to the points and shapes in the *copied* subpaths
            for i, subpath in enumerate(copied_subpaths):
                if isinstance(subpath, list): # Point list subpath
                    # Transform each point in the list
                    new_points = []
                    for p_idx in range(len(subpath)):
                        px, py = subpath[p_idx]
                        # Apply transform using Kivy's Matrix method
                        tx, ty, _ = transform.transform_point(px, py, 0) # Ignore Z
                        new_points.append((tx, ty))
                    copied_subpaths[i] = new_points # Replace list in copied_subpaths

                elif isinstance(subpath, dict) and subpath.get('type') == 'roundRect':
                    # Transforming roundRect parameters is non-trivial, especially radii
                    # under non-uniform scale, skew, or rotation.
                    # Simplification: Transform the top-left corner (x, y).
                    # Keep original width, height, and radii, but warn that they might
                    # be visually incorrect if the transform isn't just translation.
                    params = subpath # Get the dictionary
                    ox, oy = params['x'], params['y']
                    ow, oh = params['w'], params['h']
                    # Transform the origin point
                    tx, ty, _ = transform.transform_point(ox, oy, 0)
                    params['x'], params['y'] = tx, ty
                    # TODO: Implement more accurate roundRect transformation if needed.
                    # This would involve transforming all 4 corner points and potentially
                    # recalculating radii based on the transformed shape, which is complex.
                    warnings.warn("Transforming roundRect in addPath only adjusts position (x,y). Size and radii are not accurately transformed for non-translation matrices.", stacklevel=3)
                    copied_subpaths[i] = params # Update the dict in the copied list
                # Add handling for other potential shape types (e.g., 'arc') if added later

        # Extend the current path's subpaths list with the transformed copies
        self.subpaths.extend(copied_subpaths)

        # Spec: addPath creates a break in the path sequence.
        # Reset the current subpath state, so subsequent commands start fresh.
        self._reset_current_point_subpath()

class ImageData:
    """Represents raw pixel data for a rectangular area."""
    def __init__(self, width: int | float, height: int | float, data: bytes | bytearray | None = None, *, colorSpace: str = "srgb"):
        # colorSpace is part of spec, store it but don't use it actively yet
        self.colorSpace = colorSpace
        self._texture = None # Cache for Kivy texture

        try:
             # Spec requires unsigned long for width/height, use int() and check positivity
             w_int = int(width)
             h_int = int(height)
             if w_int < 0 or h_int < 0:
                  # Spec might throw? Kivy context usually uses abs(). Let's use abs().
                  warnings.warn(f"ImageData width/height negative ({width}, {height}). Using absolute values.", stacklevel=3)
             self.width = abs(w_int)
             self.height = abs(h_int)
        except (ValueError, TypeError):
             # Handle non-numeric width/height
             self.width, self.height = 0, 0
             warnings.warn(f"Invalid dimensions for ImageData ({width}, {height}). Setting to 0x0.", stacklevel=2)

        # Handle data buffer
        expected_len = self.width * self.height * 4
        if data is None:
             # If no data provided, create blank (transparent black) buffer of correct size
             self.data = b'\x00\x00\x00\x00' * (self.width * self.height) if self.width > 0 and self.height > 0 else b''
        elif isinstance(data, (bytes, bytearray)):
             # Ensure data is bytes type
             self.data = bytes(data)
             # Validate data length against dimensions
             if len(self.data) != expected_len:
                 # Spec: Throw if length mismatch
                 raise ValueError(f"ImageData data length ({len(self.data)}) does not match dimensions ({self.width}x{self.height} = {expected_len} bytes).")
        else:
             raise TypeError(f"ImageData data must be bytes, bytearray, or None, got {type(data)}")

        # Ensure data is empty if dimensions are zero
        if self.width == 0 or self.height == 0:
            self.data = b''

    @property
    def texture(self) -> Texture | None:
        """Gets a Kivy Texture for the ImageData. Creates/updates if needed.
           Texture origin is bottom-left. Data is flipped vertically during blit
           so the texture visually matches the top-left origin ImageData."""

        # Check if cached texture exists, is allocated, and matches current dimensions
        if self._texture:
            if self._texture.width == self.width and self._texture.height == self.height:
                # Texture exists and size matches.
                # WARNING: If self.data (the byte buffer) is modified externally AFTER
                # texture creation, the texture won't automatically update.
                # A mechanism to re-blit might be needed if ImageData is mutable.
                return self._texture
            else:
                # Size mismatch, invalidate cached texture
                # Logger.debug("ImageData: Cached texture size mismatch, invalidating.")
                self._texture = None

        # Create or recreate texture if dimensions are valid and data exists
        if self.width > 0 and self.height > 0 and self.data:
            expected_len = self.width * self.height * 4
            if len(self.data) == expected_len:
                try:
                    # Logger.debug(f"ImageData: Creating new texture {self.width}x{self.height}")
                    tex = Texture.create(size=(self.width, self.height), colorfmt='rgba', bufferfmt='ubyte')
                    # Blit buffer: ImageData origin is TL, Kivy Texture origin is BL.
                    # Blitting directly makes Kivy texture visually upside-down relative to ImageData.
                    tex.blit_buffer(self.data, colorfmt='rgba', bufferfmt='ubyte')
                    # Flip the texture vertically after blitting so it visually matches
                    # the top-left orientation of the ImageData when drawn normally in Kivy.
                    tex.flip_vertical()
                    self._texture = tex # Cache the new texture
                    return self._texture
                except Exception as e:
                    warnings.warn(f"Failed to create texture for ImageData ({self.width}x{self.height}): {e}", stacklevel=2)
                    self._texture = None # Ensure cache is cleared on failure
                    return None
            else:
                # Data length mismatch (Should have been caught in __init__, but check again)
                warnings.warn(f"ImageData.texture: Data length mismatch ({len(self.data)} vs {expected_len}). Cannot create texture.", stacklevel=2)
                return None
        else:
            # Width/Height is zero or no data
            # Logger.debug("ImageData: Cannot create texture, zero dimensions or no data.")
            return None

    def __repr__(self):
         status = "Empty"
         if self.width > 0 and self.height > 0:
             status = f"Has data={len(self.data) > 0}, Texture={'Allocated' if self._texture else 'None'}"
         return f"<ImageData width={self.width} height={self.height} ({status}) colorSpace='{self.colorSpace}'>"

class CanvasGradient:
    """Base class for Kivy Canvas gradients."""
    _is_kivy_gradient = True # Marker attribute for type checking

    def __init__(self):
        # Stores lists: [offset_float, (r, g, b, a)]
        self.color_stops: list[list[float, tuple[float, float, float, float]]] = []

    def addColorStop(self, offset: float, color: str | tuple | list) -> None:
        """Adds a color stop to the gradient."""
        try:
            # Spec: Validate offset (IndexSizeError for out of range 0-1)
            offset_float = float(offset)
            if not (0.0 <= offset_float <= 1.0):
                # Simulate DOM IndexSizeError with ValueError in Python
                raise ValueError(f"Offset {offset_float} must be between 0.0 and 1.0")

            # Spec: Validate color (SyntaxError if parsing fails)
            parsed_color = CSSColorParser.parse_color(color)
            if parsed_color is None:
                # Simulate DOM SyntaxError with ValueError/TypeError in Python
                raise ValueError(f"Invalid color value: '{color}'")

            # Add the stop as [offset, (r, g, b, a)]
            self.color_stops.append([offset_float, parsed_color])

            # Keep stops sorted by offset. Crucial for interpolation logic.
            # Use sort() which modifies the list in-place.
            self.color_stops.sort(key=lambda stop: stop[0])

        except (ValueError, TypeError) as e:
             # Re-raise errors caught above (offset range, color parse)
             # or other conversion errors (e.g., float(offset)).
             # This mimics the DOM exceptions behavior more closely.
             raise e # Propagate the specific error

class LinearGradient(CanvasGradient):
    """Represents a linear gradient (x0, y0) -> (x1, y1)."""
    def __init__(self, x0, y0, x1, y1):
        super().__init__()
        # Coords stored directly. Validation (finite check) happens in factory method.
        self.x0, self.y0 = float(x0), float(y0)
        self.x1, self.y1 = float(x1), float(y1)

class RadialGradient(CanvasGradient):
    """Represents a radial gradient (circle0 -> circle1)."""
    def __init__(self, x0, y0, r0, x1, y1, r1):
        super().__init__()
        # Coords/radii stored directly. Validation (finite, non-neg radius) happens in factory.
        self.x0, self.y0, self.r0 = float(x0), float(y0), float(r0)
        self.x1, self.y1, self.r1 = float(x1), float(y1), float(r1)

class ConicGradient(CanvasGradient):
    """Represents a conic gradient around (x, y) starting at angle."""
    def __init__(self, startAngle, x, y):
        super().__init__()
        # Angle in radians, center coordinates. Validation (finite) happens in factory.
        self.start_angle = float(startAngle) # Radians
        self.cx, self.cy = float(x), float(y)


# --- Shader Code Definition (FIXED) ---
# Define max color stops the shader can handle (must match shader array size!)
MAX_STOPS = 16 # Increase if more stops are frequently needed

# --- FIX START: Modify Vertex Shader ---
GRADIENT_VS = """
#ifdef GL_ES
    // Specify precision for float types in fragment shaders.
    precision highp float;
#endif

/* Inputs from Kivy Mesh */
attribute vec2 vPosition; // Vertex position in Canvas coordinates
// attribute vec4 vColor; // We don't actually use color attribute from mesh fmt
// attribute vec2 vTexCoords0; // We don't actually use texcoord attribute from mesh fmt

/* Uniforms provided by Kivy's RenderContext */
uniform mat4 modelview_mat;     // Combined Kivy widget + user transform
uniform mat4 projection_mat;    // Kivy projection matrix

/* Outputs to Fragment Shader */
varying vec2 canvas_coord; // Pass the original Canvas coordinate
varying vec4 frag_color;   // Standard Kivy varying (for linker)
varying vec2 tex_coord0;   // Standard Kivy varying (for linker)

void main() {
    // Pass the original vertex position (which is in Canvas coordinate space)
    // to the fragment shader. The FS needs this for gradient calculations.
    canvas_coord = vPosition;

    // Assign dummy/plausible values to standard Kivy varyings to satisfy the linker.
    // These values are NOT used by the Gradient Fragment Shader's logic.
    frag_color = vec4(1.0, 1.0, 1.0, 1.0); // Pass white color (or could use vColor if provided)
    tex_coord0 = vPosition * 0.01; // Pass scaled position as dummy UV (or vTexCoords0 if provided)

    // Transform the vertex position for rendering using Kivy's matrices.
    gl_Position = projection_mat * modelview_mat * vec4(vPosition.xy, 0.0, 1.0);
}
"""
# --- FIX END: Modify Vertex Shader ---

# --- FIX START: Modify Fragment Shader ---
GRADIENT_FS = f"""
#ifdef GL_ES
    precision highp float;
#endif

/* Constants */
#define MAX_STOPS {MAX_STOPS} // Must match Python definition
const float EPSILON = 1e-5; // Small value for float comparisons
const float TWO_PI = 6.28318530718;

/* Inputs from Vertex Shader */
varying vec2 canvas_coord; // Interpolated Canvas coordinate for this fragment
varying vec4 frag_color;   // Standard Kivy varying (declared but NOT used)
varying vec2 tex_coord0;   // Standard Kivy varying (declared but NOT used)

/* Uniforms from Python */
// General
uniform int u_gradient_type; // 0: linear, 1: radial, 2: conic
uniform float u_global_alpha; // Global alpha factor (0.0 to 1.0)

// Color Stops (ensure Python sends exactly MAX_STOPS values, padded if needed)
uniform int u_num_color_stops;  // Actual number of stops used (<= MAX_STOPS)
uniform float u_offsets[MAX_STOPS]; // Offsets (0.0 to 1.0, sorted)
uniform vec4 u_colors[MAX_STOPS];   // Colors (rgba) corresponding to offsets

// Linear Gradient Params
uniform vec2 u_linear_start; // (x0, y0) in Canvas coords
uniform vec2 u_linear_end;   // (x1, y1) in Canvas coords

// Radial Gradient Params
uniform vec2 u_radial_c0;    // Center 0 (x0, y0)
uniform float u_radial_r0;   // Radius 0 (r0 >= 0)
uniform vec2 u_radial_c1;    // Center 1 (x1, y1)
uniform float u_radial_r1;   // Radius 1 (r1 >= 0)

// Conic Gradient Params
uniform float u_conic_angle; // Start angle (radians)
uniform vec2 u_conic_center; // Center (cx, cy) in Canvas coords

// --- Helper: Interpolate Color ---
// (Function remains the same)
vec4 getColorAtOffset(float t) {{
    if (u_num_color_stops <= 0) {{
        return vec4(0.0, 0.0, 0.0, 0.0); // Transparent black if no stops
    }}
    if (u_num_color_stops == 1) {{
        return u_colors[0]; // Return the only color
    }}
    t = clamp(t, 0.0, 1.0);
    if (t <= u_offsets[0] + EPSILON) {{
        return u_colors[0];
    }}
    // Note: Array index is u_num_color_stops - 1 for the last valid stop
    if (t >= u_offsets[u_num_color_stops - 1] - EPSILON) {{
        return u_colors[u_num_color_stops - 1];
    }}
    // Loop up to the second-to-last stop (index u_num_color_stops - 2)
    for (int i = 0; i < MAX_STOPS - 1; ++i) {{
        // Optimization/safety: Break if we've passed the actual number of stops
        if (i >= u_num_color_stops - 1) break;

        // Check if t is within the interval [offset[i], offset[i+1]]
        if (t >= u_offsets[i] && t <= u_offsets[i+1]) {{
            float offset1 = u_offsets[i];
            float offset2 = u_offsets[i+1];
            vec4 color1 = u_colors[i];
            vec4 color2 = u_colors[i+1];
            float range = offset2 - offset1;
            // Avoid division by zero if stops have the same offset
            if (range < EPSILON) {{
                return color2; // Return color of the latter stop
            }}
            float factor = (t - offset1) / range;
            return mix(color1, color2, factor); // Linear interpolation (GLSL mix)
        }}
    }}
    // Fallback: Should ideally be unreachable if checks above are correct.
    return u_colors[u_num_color_stops - 1];
}}

// --- Helper: Linear Gradient Offset 't' ---
// (Function remains the same)
float getLinearOffset(vec2 p) {{
    vec2 P0 = u_linear_start;
    vec2 P1 = u_linear_end;
    vec2 diff = P1 - P0;
    float len_sq = dot(diff, diff);
    if (len_sq < EPSILON * EPSILON) {{
        return 1.0; // End color stop
    }}
    vec2 rel_p = p - P0;
    float t = dot(rel_p, diff) / len_sq;
    return clamp(t, 0.0, 1.0);
}}

// --- Helper: Radial Gradient Offset 't' ---
// (Function remains the same)
float getRadialOffset(vec2 p) {{
    vec2 C0 = u_radial_c0; float R0 = u_radial_r0;
    vec2 C1 = u_radial_c1; float R1 = u_radial_r1;
    vec2 dC = C1 - C0;
    float dR = R1 - R0;
    vec2 p_rel = p - C0;
    float a = dot(dC, dC) - dR * dR;
    float b = 2.0 * (dot(p_rel, dC) - R0 * dR);
    float c = dot(p_rel, p_rel) - R0 * R0;
    float t = 1.0; // Default to end color stop
    if (dot(dC, dC) < EPSILON * EPSILON) {{ // Concentric circles
        if (abs(dR) < EPSILON) {{ // R0 == R1 and C0 == C1
            t = 1.0;
        }} else {{ // Concentric circles with different radii
            float dist_p_c0 = length(p_rel);
            t = (dist_p_c0 - R0) / dR;
        }}
    }}
    else if (abs(a) < EPSILON) {{ // Degenerate case: linear equation
        if (abs(b) < EPSILON) {{ // Gradient ill-defined
            t = 1.0;
        }} else {{
            t = -c / b;
        }}
    }} else {{ // Quadratic case
        float discriminant = b * b - 4.0 * a * c;
        if (discriminant < 0.0) {{ // Outside gradient influence
            t = 1.0;
        }} else {{ // Use the larger root (outer intersection)
            float sqrt_discriminant = sqrt(discriminant);
            float t1 = (-b + sqrt_discriminant) / (2.0 * a);
            t = t1;
        }}
    }}
    return clamp(t, 0.0, 1.0);
}}


// --- Helper: Conic Gradient Offset 't' ---
// (Function remains the same)
float getConicOffset(vec2 p) {{
    vec2 Center = u_conic_center;
    float startAngle = u_conic_angle; // Input angle in radians
    vec2 D = p - Center;
    if (length(D) < EPSILON) {{ // Point is at the center
        return 0.0; // Use color of the first stop
    }}
    // Calculate angle of the vector D using atan2(y, x). Result is in [-PI, PI].
    float angle = atan(D.y, D.x); // GLSL atan(y, x)
    // Normalize calculated angle to [0, 2*PI) range.
    if (angle < 0.0) {{
        angle += TWO_PI;
    }}
    // Normalize startAngle to [0, 2*PI) as well.
    float normalizedStartAngle = startAngle;
    while (normalizedStartAngle < 0.0) {{ normalizedStartAngle += TWO_PI; }}
    normalizedStartAngle = mod(normalizedStartAngle, TWO_PI);
    // Calculate the angular difference relative to the normalized start angle.
    float offset_angle = angle - normalizedStartAngle;
    // Normalize the difference to [0, 2*PI) range again.
    if (offset_angle < 0.0) {{
        offset_angle += TWO_PI;
    }}
    offset_angle = mod(offset_angle, TWO_PI);
    // Final offset t = normalized angle difference / (2 * PI) -> maps to [0, 1)
    float t = offset_angle / TWO_PI;
    return t; // Return the calculated offset (should be in [0, 1) range)
}}


void main() {{
    float t = 0.0; // Gradient offset parameter [0, 1]

    // Calculate offset 't' based on gradient type and fragment's canvas coordinate
    if (u_gradient_type == 0) {{ // Linear
        t = getLinearOffset(canvas_coord);
    }} else if (u_gradient_type == 1) {{ // Radial
        t = getRadialOffset(canvas_coord);
    }} else if (u_gradient_type == 2) {{ // Conic
        t = getConicOffset(canvas_coord);
    }} else {{
        // Unknown gradient type, output transparent black
        // Note: We don't use the input varying 'frag_color' here
        gl_FragColor = vec4(0.0, 0.0, 0.0, 0.0);
        return;
    }}

    // Get the interpolated color based on the calculated offset 't' and color stops
    vec4 gradient_color = getColorAtOffset(t);

    // Apply global alpha multiplicativeley
    gradient_color.a *= u_global_alpha;

    // Apply premultiplied alpha: color.rgb = color.rgb * color.a
    gradient_color.rgb *= gradient_color.a;

    // Assign the final calculated color. We don't use input 'frag_color'.
    gl_FragColor = gradient_color;
}}
"""
# --- FIX END: Modify Fragment Shader ---


class Canvas2DContext(Widget):
    # Default state values (constants)
    DEFAULT_FILL_STYLE = (0.0, 0.0, 0.0, 1.0) # black
    DEFAULT_STROKE_STYLE = (0.0, 0.0, 0.0, 1.0) # black
    DEFAULT_LINE_WIDTH = 1.0
    DEFAULT_LINE_CAP = 'butt'       # 'butt', 'round', 'square'
    DEFAULT_LINE_JOIN = 'miter'     # 'round', 'bevel', 'miter'
    DEFAULT_MITER_LIMIT = 10.0
    DEFAULT_GLOBAL_ALPHA = 1.0
    DEFAULT_FONT = '10px sans-serif' # Spec default
    DEFAULT_TEXT_ALIGN = 'start'    # 'left','right','center','start','end'
    DEFAULT_TEXT_BASELINE = 'alphabetic' # 'top','hanging','middle','alphabetic','ideographic','bottom'
    DEFAULT_DIRECTION = 'ltr'       # 'ltr', 'rtl', 'inherit'
    DEFAULT_IMAGE_SMOOTHING = True
    DEFAULT_IMAGE_SMOOTHING_QUALITY = 'low' # 'low'|'medium'|'high'
    DEFAULT_FILTER = 'none'
    DEFAULT_COMPOSITE_OP = 'source-over'
    DEFAULT_SHADOW_BLUR = 0.0
    DEFAULT_SHADOW_COLOR = (0.0, 0.0, 0.0, 0.0) # transparent black
    DEFAULT_SHADOW_OFFSET_X = 0.0
    DEFAULT_SHADOW_OFFSET_Y = 0.0

    # --- Shader Related Setup ---
    # Kivy Mesh vertex format for shader-based gradient rendering.
    # We only need the raw 2D position in Canvas coordinates.
    _gradient_vertex_format_shader = [
        (b'vPosition', 2, 'float'), # Canvas coordinates (x, y) sent to VS
    ]

    # Shader code strings (FIXED ABOVE)
    _gradient_vs = GRADIENT_VS
    _gradient_fs = GRADIENT_FS
    _MAX_COLOR_STOPS = MAX_STOPS # Get constant from shader definition module level

    def __init__(self, **kwargs):
        # --- RenderContext for Gradient Shader ---
        # Create a single RenderContext instance for drawing all gradients.
        # use_parent_projection/modelview=True tells Kivy to supply matrices based
        # on the widget's hierarchy and the active MatrixInstruction.
        self._render_context = None # Initialize to None
        try:
            rc = RenderContext(use_parent_projection=True, use_parent_modelview=True)
            # Assign FIXED shader code
            rc.shader.vs = self._gradient_vs
            rc.shader.fs = self._gradient_fs
            # Check if shader compiled and linked successfully
            # Kivy's Shader object has a 'success' property after setting vs/fs
            if not rc.shader.success:
                 # Shader loading might log errors, let's add one too
                 log_info = rc.shader.get_program_log() # Get link/compile log
                 Logger.error(f"Canvas2DContext: Failed to compile/link gradient shader. Log:\n{log_info}")
                 # Fallback? For now, store None. Gradient rendering will fail.
                 self._render_context = None # Explicitly set to None on failure
            else:
                 self._render_context = rc
                 Logger.info("Canvas2DContext: Gradient shader initialized successfully.")

            # Alternative: Load from .glsl files (make sure files contain fixed shader code)
            # shader_path = os.path.join(os.path.dirname(__file__), 'shaders')
            # self._render_context.shader.vs = open(os.path.join(shader_path, 'gradient.vs')).read()
            # self._render_context.shader.fs = open(os.path.join(shader_path, 'gradient.fs')).read()

        except Exception as e:
            Logger.error(f"Canvas2DContext: Failed to initialize shader RenderContext: {e}", exc_info=True)
            # Ensure _render_context remains None or handle fallback
            self._render_context = None # Ensure it's None on any exception

        # --- Standard Widget and State Initialization ---
        super().__init__(**kwargs)

        self._fbo: Fbo | None = None # For getImageData / potentially other offscreen ops
        self._texture_cache: dict[str, Texture | TextureRegion] = {} # For loadImage (string sources)
        self._state_stack: list[dict] = [] # For save/restore

        # Transformation matrices
        self._user_matrix = Matrix() # User transform (scale, rotate, translate, transform)
        self._base_matrix = Matrix() # Base transform (Y-flip + translate for Canvas coords)
        # Required by Kivy Mesh instructions even if we don't use normals explicitly.
        self._update_normal_matrix_instr = UpdateNormalMatrix()

        # --- Canvas State Variables (Initialize with defaults) ---
        self._fillStyle: tuple | CanvasGradient = self.DEFAULT_FILL_STYLE
        self._strokeStyle: tuple | CanvasGradient = self.DEFAULT_STROKE_STYLE
        self._lineWidth: float = self.DEFAULT_LINE_WIDTH
        self._lineCap: str = self.DEFAULT_LINE_CAP
        self._lineJoin: str = self.DEFAULT_LINE_JOIN
        self._miterLimit: float = self.DEFAULT_MITER_LIMIT
        self._lineDash: list[float] = []
        self._lineDashOffset: float = 0.0
        self._globalAlpha: float = self.DEFAULT_GLOBAL_ALPHA
        self._globalCompositeOperation: str = self.DEFAULT_COMPOSITE_OP
        self._imageSmoothingEnabled: bool = self.DEFAULT_IMAGE_SMOOTHING
        self._imageSmoothingQuality: str = self.DEFAULT_IMAGE_SMOOTHING_QUALITY
        self._filter: str = self.DEFAULT_FILTER # Not implemented visually
        self._font: str = self.DEFAULT_FONT # CSS font string
        self._textAlign: str = self.DEFAULT_TEXT_ALIGN
        self._textBaseline: str = self.DEFAULT_TEXT_BASELINE
        self._direction: str = self.DEFAULT_DIRECTION
        self._shadowBlur: float = self.DEFAULT_SHADOW_BLUR # Not implemented visually
        self._shadowColor: tuple = self.DEFAULT_SHADOW_COLOR # Not implemented visually
        self._shadowOffsetX: float = self.DEFAULT_SHADOW_OFFSET_X # Not implemented visually
        self._shadowOffsetY: float = self.DEFAULT_SHADOW_OFFSET_Y # Not implemented visually

        # Internal derived font properties (updated when self.font is set)
        self._font_helper = CSSFont() # Initialize helper
        # --- FIX: Ensure _update_derived_font_properties is called AFTER super().__init__ ---
        # super().__init__(**kwargs) # Moved up

        # Path state
        self._current_path = Path2D() # The path being actively built
        self._clip_path: Path2D | None = None # Stores the Path2D used for the current clip region
        self._clip_fill_rule: str = 'nonzero' # Fill rule used for the active clip path

        # *** 新增：背景设置 ***
        with self.canvas.before:
            # 设置背景色为白色 (r=1, g=1, b=1, a=1)
            self._background_color = Color(1, 1, 1, 1)
            # 创建一个覆盖整个 Widget 区域的矩形
            self._background_rect = Rectangle(pos=self.pos, size=self.size)

        # 绑定 pos 和 size 的变化来更新背景矩形
        self.bind(pos=self._update_background, size=self._update_background)

        # --- Final Setup ---
        # Set initial state and Kivy canvas instructions (calls reset())
        self.reset()
        # Bind position/size changes to update base transform and clear FBO
        self.bind(pos=self._on_pos_size_change, size=self._on_pos_size_change)
        # Initial setup call to set the base matrix correctly based on initial size
        self._on_pos_size_change()
        # --- *** 修改点 1: 延迟第一次字体属性更新 *** ---
        # REMOVED: self._update_derived_font_properties()
        # Schedule the first update for the next frame
        Clock.schedule_once(self._update_derived_font_properties, 0)
        # self._update_background() # Call initial update for background (reset should handle this too)

    # *** 新增：背景更新方法 ***
    def _update_background(self, instance=None, value=None):
        """当 Widget 的位置或大小改变时，更新背景矩形。"""
        # 确保 _background_rect 已经被创建（在 __init__ 中）
        if hasattr(self, '_background_rect'):
            self._background_rect.pos = self.pos
            self._background_rect.size = self.size


    def _update_derived_font_properties(self, *args): # Add *args for Clock compatibility
        """Updates internal font properties based on the _font_helper state."""
        # This method is now safe to call after initialization or when font changes
        old_name = getattr(self, '_font_name', None) # Check if exists before comparing
        self._font_size = self._font_helper.font_size_px
        self._font_name = self._font_helper.kivy_font_name # This now calls LabelBase check safely
        self._font_italic = self._font_helper.is_italic
        self._font_bold = self._font_helper.is_bold
        # Optional: Log if the derived font name actually changed
        if old_name is not None and old_name != self._font_name:
            Logger.debug(f"Canvas Font: Derived Kivy font name updated to '{self._font_name}' for CSS font '{self._font}'")

    def _on_pos_size_change(self, *args):
        """Callback when widget position or size changes."""
        # Update the base matrix which handles Y-flipping based on height
        self._update_base_matrix()
        # Invalidate FBO used for getImageData, as size/content is now stale
        # FBO needs recreation if size changes, clearing alone isn't enough.
        # Let getImageData handle recreation when needed. Set to None for now.
        if self._fbo:
             # Logger.debug("Canvas size changed, invalidating FBO.")
             # Should we delete the old FBO texture? Kivy might GC it.
             self._fbo = None
        # *** 同时更新背景（虽然bind也会触发，但直接调用更明确） ***
        self._update_background()

    def _update_base_matrix(self):
        """Sets the base transformation matrix: Canvas TL origin -> Kivy BL origin."""
        # Use max(1, size) to avoid issues if size is 0 during init or resize
        w = max(1, self.size[0])
        h = max(1, self.size[1])
        # Transformation: Scale Y by -1 (flip), then translate Y up by height.
        # Maps Canvas (x, y) to Kivy (x, h - y) before user transform.
        
        self._base_matrix = Matrix().scale(x=1, y=-1, z=1).translate(x=0, y=h, z=0)

    def _setup_canvas(self):
        """Clears and sets up the basic Kivy canvas instructions needed by context."""
        # 注意：这里只清空 self.canvas，不会影响 self.canvas.before
        self.canvas.clear()
        # Mesh instructions require UpdateNormalMatrix to be present in the canvas
        # even if normals aren't used. Add it once permanently.
        self.canvas.add(self._update_normal_matrix_instr)
        # The RenderContext for gradients is added temporarily during gradient draws.
        
    # --- Public API Properties (Setters/Getters for Canvas State) ---

    @property
    def fillStyle(self): return self._fillStyle
    @fillStyle.setter
    def fillStyle(self, value):
        # Check for Kivy gradient object first
        if hasattr(value, '_is_kivy_gradient') and value._is_kivy_gradient:
            self._fillStyle = value # Store the gradient object
        elif isinstance(value, (str, tuple, list)):
            # Try parsing as CSS color string
            parsed_color = CSSColorParser.parse_color(value)
            if parsed_color is not None:
                self._fillStyle = parsed_color # Store RGBA tuple (0-1 floats)
            # else: Spec says ignore invalid values, keep the old one. Do nothing.
        # else: Ignore other types (Pattern object would go here later)
            # warnings.warn(f"Unsupported fillStyle type: {type(value)}.", stacklevel=2)

    @property
    def strokeStyle(self): return self._strokeStyle
    @strokeStyle.setter
    def strokeStyle(self, value):
        if hasattr(value, '_is_kivy_gradient') and value._is_kivy_gradient:
            self._strokeStyle = value
            # Warn that gradient strokes are not fully implemented visually
            warnings.warn("Gradient set as strokeStyle, but rendering will fallback to solid color.", stacklevel=2)
        elif isinstance(value, (str, tuple, list)):
            parsed_color = CSSColorParser.parse_color(value)
            if parsed_color is not None:
                self._strokeStyle = parsed_color
        # else: Ignore other types (Pattern)

    # --- Line Style Properties ---
    @property
    def lineWidth(self) -> float: return self._lineWidth
    @lineWidth.setter
    def lineWidth(self, value):
        try:
             lw = float(value)
             # Spec: If 0, negative, infinite, or NaN, value is ignored.
             if lw > 0 and math.isfinite(lw):
                 self._lineWidth = lw
        except (ValueError, TypeError): pass # Ignore non-numeric or conversion errors

    @property
    def lineCap(self) -> str: return self._lineCap
    @lineCap.setter
    def lineCap(self, value: str):
        # Spec: If value is not one of the keywords, ignore it.
        val = str(value).lower()
        if val in ('butt', 'round', 'square'):
             self._lineCap = val

    @property
    def lineJoin(self) -> str: return self._lineJoin
    @lineJoin.setter
    def lineJoin(self, value: str):
        val = str(value).lower()
        if val in ('round', 'bevel', 'miter'):
             self._lineJoin = val

    @property
    def miterLimit(self) -> float: return self._miterLimit
    @miterLimit.setter
    def miterLimit(self, value):
        try:
             ml = float(value)
             # Spec: If 0, negative, infinite, or NaN, value is ignored.
             if ml > 0 and math.isfinite(ml):
                 self._miterLimit = ml
        except (ValueError, TypeError): pass # Ignore

    def setLineDash(self, segments: list | tuple) -> None:
        """Sets the line dash pattern."""
        try:
            # Ensure input is iterable, try converting if not list/tuple
            if not isinstance(segments, (list, tuple)):
                 # Attempt to iterate (will raise TypeError if not possible)
                 segments = list(segments)

            validated_segments = []
            for x in segments:
                 val = float(x)
                 # Spec: If any value is negative, infinite, or NaN, throw TypeError.
                 if not (math.isfinite(val) and val >= 0):
                     # Raise TypeError consistent with spec behavior
                     raise TypeError(f"Segment value {val} is invalid (must be finite and non-negative).")
                 validated_segments.append(val)

            # Spec: If the number of segments is odd, the list is duplicated.
            if len(validated_segments) % 2 != 0:
                validated_segments.extend(validated_segments) # Duplicate the list in-place

            # If list becomes empty after duplication (e.g., input []), store empty.
            self._lineDash = validated_segments

            # Warn if dashing is set, as Kivy's Line doesn't natively support it easily.
            if self._lineDash:
                warnings.warn("setLineDash() called, but Kivy's Line instruction does not natively support dashed lines visually.", stacklevel=2)

        except (ValueError, TypeError) as e:
             # Reraise TypeError for validation failures, propagate others.
             raise TypeError(f"Invalid segments sequence for setLineDash: {e}") from e

    def getLineDash(self) -> list[float]:
        """Returns a copy of the current line dash list."""
        return self._lineDash[:] # Return a copy to prevent external modification

    @property
    def lineDashOffset(self) -> float: return self._lineDashOffset
    @lineDashOffset.setter
    def lineDashOffset(self, value: float):
        try:
            offset = float(value)
            # Spec: If infinite or NaN, value is ignored.
            if math.isfinite(offset):
                 self._lineDashOffset = offset
                 # Warn if offset is set while dashing is active (and not supported)
                 if self._lineDash:
                     warnings.warn("lineDashOffset set, but Kivy's Line instruction does not support dashing.", stacklevel=2)
        except (ValueError, TypeError): pass # Ignore non-numeric

    # --- Text Properties ---
    @property
    def font(self) -> str: return self._font
    @font.setter
    def font(self, value: str):
        font_str = str(value)
        # Try parsing the new font string
        if self._font_helper.parse_font_str(font_str):
            # If parsing succeeded, update internal state
            self._font = self._font_helper.font_str
            # --- *** 修改点 3: 保持这里的更新调用 *** ---
            # This is called *after* init, so LabelBase should be ready
            self._update_derived_font_properties()
        # else: Parsing failed (warning issued inside helper), keep old state.

    @property
    def textAlign(self) -> str: return self._textAlign
    @textAlign.setter
    def textAlign(self, value: str):
        val = str(value).lower()
        if val in ('left', 'right', 'center', 'start', 'end'):
             self._textAlign = val

    @property
    def textBaseline(self) -> str: return self._textBaseline
    @textBaseline.setter
    def textBaseline(self, value: str):
        val = str(value).lower()
        if val in ('top', 'hanging', 'middle', 'alphabetic', 'ideographic', 'bottom'):
             self._textBaseline = val

    @property
    def direction(self) -> str: return self._direction
    @direction.setter
    def direction(self, value: str):
        val = str(value).lower()
        if val in ('ltr', 'rtl', 'inherit'):
            self._direction = val
            # Warn if RTL is set, as it only affects start/end alignment here
            if val != 'ltr':
                warnings.warn(f"Text direction '{val}' set. Only affects textAlign='start'/'end'. Full RTL layout not implemented.", stacklevel=2)

    # --- Compositing & Alpha ---
    @property
    def globalAlpha(self) -> float: return self._globalAlpha
    @globalAlpha.setter
    def globalAlpha(self, value):
        try:
            alpha = float(value)
            # Spec: If infinite or NaN, ignore. Clamp to [0, 1].
            if math.isfinite(alpha):
                 self._globalAlpha = max(0.0, min(1.0, alpha))
        except (ValueError, TypeError): pass # Ignore

    @property
    def globalCompositeOperation(self) -> str: return self._globalCompositeOperation
    @globalCompositeOperation.setter
    def globalCompositeOperation(self, value: str):
        gco = str(value).lower()
        # List of valid W3C spec composite operations
        valid_ops = [
            'source-over', 'source-in', 'source-out', 'source-atop',
            'destination-over', 'destination-in', 'destination-out', 'destination-atop',
            'lighter', 'copy', 'xor', 'multiply', 'screen', 'overlay', 'darken',
            'lighten', 'color-dodge', 'color-burn', 'hard-light', 'soft-light',
            'difference', 'exclusion', 'hue', 'saturation', 'color', 'luminosity'
        ]
        if gco in valid_ops:
            self._globalCompositeOperation = gco
            # Kivy's default blend func IS 'source-over'. Others require specific
            # glBlendFunc/glBlendEquation setups, which are complex to manage per-draw.
            # TODO: Implement other blend modes if needed via custom RenderContext or Callbacks.
            if gco != 'source-over':
                warnings.warn(f"globalCompositeOperation '{gco}' set, but Kivy implementation currently only supports 'source-over' visual effect via default OpenGL blend mode.", stacklevel=2)

    # --- Filtering & Smoothing ---
    @property
    def filter(self) -> str: return self._filter
    @filter.setter
    def filter(self, value: str):
        # Canvas spec filter is complex ('none' or <filter-function-list>)
        # We only store the string, no visual implementation yet.
        # TODO: Implement filters (e.g., blur, contrast) possibly using shaders.
        self._filter = str(value)
        if self._filter != 'none':
            warnings.warn(f"Canvas filter '{self._filter}' set but visual effects are not implemented.", stacklevel=2)

    @property
    def imageSmoothingEnabled(self) -> bool: return self._imageSmoothingEnabled
    @imageSmoothingEnabled.setter
    def imageSmoothingEnabled(self, value: bool):
        # Basic type check/conversion
        self._imageSmoothingEnabled = bool(value)

    @property
    def imageSmoothingQuality(self) -> str: return self._imageSmoothingQuality
    @imageSmoothingQuality.setter
    def imageSmoothingQuality(self, value: str):
        val = str(value).lower()
        if val in ('low', 'medium', 'high'):
             self._imageSmoothingQuality = val

    # --- Shadow Properties (State only, no visual effect) ---
    # TODO: Implement shadows, likely using FBOs and blur shaders.
    @property
    def shadowBlur(self) -> float: return self._shadowBlur
    @shadowBlur.setter
    def shadowBlur(self, value):
        try:
            blur = float(value)
            # Spec: If negative, infinite, or NaN, ignore.
            if math.isfinite(blur) and blur >= 0:
                 if self._shadowBlur != blur and blur > 0:
                      warnings.warn("shadowBlur set but shadow effects are not implemented.", stacklevel=2)
                 self._shadowBlur = blur
        except (ValueError, TypeError): pass # Ignore

    @property
    def shadowColor(self) -> str: # Spec returns DOMString (CSS color)
        # Convert internal (r,g,b,a) tuple back to rgba string for spec compliance.
        r, g, b, a = self._shadowColor
        # Format with integer values 0-255 for RGB, float for A (clamp values just in case).
        r_int = max(0, min(255, int(r*255 + 0.5)))
        g_int = max(0, min(255, int(g*255 + 0.5)))
        b_int = max(0, min(255, int(b*255 + 0.5)))
        a_float = max(0.0, min(1.0, a))
        return f"rgba({r_int}, {g_int}, {b_int}, {a_float})"

    @shadowColor.setter
    def shadowColor(self, value):
        # Spec: If color cannot be parsed as CSS color, ignore.
        parsed_color = CSSColorParser.parse_color(value)
        if parsed_color:
             if self._shadowColor != parsed_color and parsed_color[3] > 0 and self._shadowBlur > 0:
                 warnings.warn("shadowColor/shadowBlur set but shadow effects are not implemented.", stacklevel=2)
             self._shadowColor = parsed_color # Store as (r,g,b,a) tuple
        # else: ignore invalid color string

    @property
    def shadowOffsetX(self) -> float: return self._shadowOffsetX
    @shadowOffsetX.setter
    def shadowOffsetX(self, value):
        try:
            offset = float(value)
            # Spec: If infinite or NaN, ignore.
            if math.isfinite(offset):
                 if self._shadowOffsetX != offset and offset != 0 and self._shadowBlur > 0 and self._shadowColor[3] > 0:
                     warnings.warn("shadowOffset/shadowBlur set but shadow effects are not implemented.", stacklevel=2)
                 self._shadowOffsetX = offset
        except (ValueError, TypeError): pass # Ignore

    @property
    def shadowOffsetY(self) -> float: return self._shadowOffsetY
    @shadowOffsetY.setter
    def shadowOffsetY(self, value):
        try:
            offset = float(value)
            # Spec: If infinite or NaN, ignore.
            if math.isfinite(offset):
                 if self._shadowOffsetY != offset and offset != 0 and self._shadowBlur > 0 and self._shadowColor[3] > 0:
                      warnings.warn("shadowOffset/shadowBlur set but shadow effects are not implemented.", stacklevel=2)
                 self._shadowOffsetY = offset
        except (ValueError, TypeError): pass # Ignore


    # --- Helper Methods ---

    def _get_current_color(self, style_type: str) -> tuple[float, float, float, float] | None:
        """Gets RGBA tuple for fill or stroke style IF it's a solid color.
           Applies globalAlpha. Returns None if style is gradient/pattern or invalid."""
        base_style = self._fillStyle if style_type == 'fill' else self._strokeStyle

        # Check if it's a solid color tuple/list
        if isinstance(base_style, (tuple, list)) and len(base_style) == 4:
             # Assume tuple contains valid 0-1 float values
             r, g, b, a = base_style
             # Apply globalAlpha multiplicatively
             final_a = max(0.0, min(1.0, a * self._globalAlpha))
             return (r, g, b, final_a)
        else:
             # Style is a gradient, pattern, or invalid. Return None.
             # Caller should handle the None case (e.g., skip drawing or use fallback).
             # Avoid warning spam here, let caller decide.
             return None

    def _get_smoothing_filter(self) -> str:
        """Determines Kivy texture filter ('nearest' or 'linear') based on imageSmoothing settings."""
        if not self._imageSmoothingEnabled:
            return 'nearest'
        else:
            # Kivy only has 'nearest' and 'linear'. Map quality levels.
            # 'low' -> linear, 'medium'/'high' -> linear (no mipmap distinction here)
            # Could potentially use mipmap filters if textures support them.
            return 'linear' # Use linear for all smoothed qualities

    def _push_drawing_state(self):
        """Saves Kivy matrix state and applies current transform & clipping."""
        self.canvas.add(PushMatrix())
        # Add the combined base * user transform instruction to the canvas.
        # This affects subsequent Kivy drawing instructions (Rectangle, Line, Mesh).
        baseInstr = MatrixInstruction()
        cloneMatrix = Matrix()
        cloneMatrix.set(flat = self._base_matrix.get())
        baseInstr.matrix = cloneMatrix
        self.canvas.add(baseInstr)
        userInstr = MatrixInstruction()
        cloneMatrix = Matrix()
        cloneMatrix.set(flat = self._user_matrix.get())
        userInstr.matrix = cloneMatrix
        self.canvas.add(userInstr)
        # Apply clipping path using stencil buffer (if a clip path is set)
        self._begin_clip()

    def _pop_drawing_state(self):
        """Removes clipping effect and restores Kivy matrix state."""
        self._end_clip() # Remove stencil clipping effect (must be before PopMatrix)
        self.canvas.add(PopMatrix()) # Restore previous Kivy matrix

    def _begin_clip(self):
        """Sets up stencil buffer operations to mask drawing based on _clip_path."""
        if not self._clip_path:
            return

        # Phase 1: Define the mask
        self.canvas.add(StencilPush())
        # Kivy internally sets glStencilOp to write/increment stencil buffer value
        # for drawing instructions between StencilPush and StencilUse.
        self.canvas.add(Color(0, 0, 0, 0)) # Draw transparent for mask generation
        # Generate geometry needed for clipping (stencil mask)
        # Use _get_geometry_for_fill as it provides mesh data suitable for stencil
        clip_geometry = self._get_geometry_for_fill(self._clip_path)
        if clip_geometry:
             # Render the Mesh geometry into the stencil buffer
             self.canvas.add(Mesh(vertices=clip_geometry['vertices'], indices=clip_geometry['indices'], mode='triangles'))
        else:
             warnings.warn("Clip path geometry could not be generated for stencil.", stacklevel=3)
             # If no geometry, stencil buffer won't be written, clip effectively fails.
             # We should probably pop the stencil state here to avoid issues?
             # StencilPop() # Balance the StencilPush if we bail early
             # Let's keep the push/use/unuse/pop structure consistent, even if mask is empty.

        # Phase 2: Use the mask
        # Kivy internally sets glStencilFunc to test against the written value
        # (typically GL_EQUAL against the reference value Kivy manages) and
        # sets glStencilOp to GL_KEEP.
        # The fillRule ('nonzero' vs 'evenodd') affects how the mask is written in Phase 1,
        # but the standard Kivy test in Phase 2 should work for nonzero.
        # EvenOdd might require manual GL calls via Callback if Kivy's default test isn't right.
        self.canvas.add(StencilUse()) # Use Kivy's default test setup

        # Note: If 'evenodd' clipping doesn't work as expected, this is the area
        # that might need manual OpenGL calls via kivy.graphics.Callback.
        if self._clip_fill_rule == 'evenodd':
            warnings.warn("EvenOdd clipping rule might not render correctly with default Kivy StencilUse. NonZero rule is better supported.", stacklevel=3)

    def _end_clip(self):
        """Disables stencil test and restores previous stencil state."""
        if self._clip_path: # Only pop/unuse if _begin_clip was called
            # Phase 3: Implicitly handled by Kivy - drawing the mask again
            # between StencilUnUse and StencilPop clears the stencil entry.
            # The StencilUnUse instruction itself likely sets glStencilOp to allow writing again.
            self.canvas.add(StencilUnUse())
            # Phase 4: Pop the stencil state
            self.canvas.add(StencilPop())

    def _render_path_geometry(self, path: Path2D, mode: str) -> None:
        """
        Internal: Renders Path2D geometry using Kivy instructions for STROKING or
        basic FILLING (primarily for stencil mask generation).
        Assumes it's called within a Kivy canvas context (`with self.canvas:`).
        Does NOT set color or apply transforms (caller handles those via state).
        """
        if not path or not path.subpaths:
            return

        # --- FIX START: Pre-calculate Kivy cap value once ---
        kivy_cap = 'none' # Default mapping for 'butt' and invalid values
        if self._lineCap == 'round':
            kivy_cap = 'round'
        elif self._lineCap == 'square':
            kivy_cap = 'square'
        # --- FIX END ---

        # This function expects to be called where instructions are added to self.canvas
        # (e.g., within `with self.canvas:` or directly).

        for subpath_item in path.subpaths:
            if isinstance(subpath_item, dict) and subpath_item.get('type') == 'roundRect':
                # Render RoundRect shape
                params = subpath_item
                pos = (params['x'], params['y'])
                size = (params['w'], params['h'])
                # Use the simplified Kivy radius list generated in Path2D.roundRect
                kivy_radii = params['kivy_radii'] # List [tl, tr, br, bl]

                if mode == 'fill':
                    # For filling stencil mask or simple solid color (if stencil not used).
                    # Kivy's RoundedRectangle generates appropriate fill geometry.
                     self.canvas.add(RoundedRectangle(pos=pos, size=size, radius=kivy_radii))
                elif mode == 'stroke':
                     # Use Kivy's Line with rounded_rectangle property for stroking outline
                     # Args: x, y, w, h, tl, tr, br, bl
                    line_args = (pos[0], pos[1], size[0], size[1], *kivy_radii)
                    line_inst = Line(rounded_rectangle=line_args,
                                     width=self._lineWidth,
                                     cap=kivy_cap,
                                     joint=self._lineJoin)
                    # if self._lineJoin == 'miter':
                         # AttributeError: 'Line' object has no attribute 'miter_limit'
                         # line_inst.miter_limit = self._miterLimit # REMOVED/COMMENTED OUT
                    self.canvas.add(line_inst)

            elif isinstance(subpath_item, list):
                # Render polygon/polyline list subpath
                point_tuples = subpath_item
                num_points = len(point_tuples)
                if not point_tuples: continue # Skip empty point lists

                if mode == 'fill' and num_points >= 3:
                    # Generate Mesh geometry for filling arbitrary polygons (for stencil mask)
                    # ... (rest of fill logic remains the same) ...
                    mesh_vertices = []
                    for x, y in point_tuples:
                        mesh_vertices.extend([x, y, 0.0, 0.0]) # Add dummy UVs
                    indices = []
                    if num_points >= 3:
                        for i in range(1, num_points - 1):
                            indices.extend([0, i, i + 1])
                    if indices:
                        self.canvas.add(Mesh(vertices=mesh_vertices, indices=indices, mode='triangles'))


                elif mode == 'stroke' and num_points >= 2:
                     # Render stroke geometry using Kivy Line for polylines
                    flat_points = [coord for point in point_tuples for coord in point]
                     # Check if the polyline subpath was explicitly closed in Path2D
                    is_closed = num_points > 2 and math.dist(point_tuples[0], point_tuples[-1]) < 1e-9

                    line_inst = Line(points=flat_points,
                                      width=self._lineWidth,
                                      cap=kivy_cap,
                                      joint=self._lineJoin,
                                      close=is_closed)
                     # if self._lineJoin == 'miter':
                         # AttributeError: 'Line' object has no attribute 'miter_limit'
                         # line_inst.miter_limit = self._miterLimit # REMOVED/COMMENTED OUT
                    self.canvas.add(line_inst)
            # else: Handle other potential subpath types (curves?) if implemented later


    def _get_geometry_for_fill(self, target_path: Path2D) -> dict | None:
        """
        Generates vertices (in Canvas coordinates) and indices suitable for
        a Kivy Mesh instruction to fill the given Path2D.
        Used primarily for gradient fills where a single Mesh is needed for the shader.
        Returns {'vertices': [x1,y1,x2,y2...], 'indices': [i1,i2,i3...]} or None.

        Uses simple triangulation (triangle fan for polygons, rect approx for roundRect).
        WARNING: May produce incorrect geometry for concave or self-intersecting paths.
        Vertices are pairs (x,y), but flattened to [x1, y1, x2, y2, ...] for Kivy Mesh.
        Indices refer to the vertex number (0, 1, 2, ...).
        """
        if not target_path or not target_path.subpaths:
            return None

        all_vertices = [] # Flat list [x1, y1, x2, y2, ...] for Mesh format
        all_indices = []
        vertex_offset = 0 # Track starting vertex index for each subpath's indices

        for subpath_item in target_path.subpaths:
            sub_vertices = [] # Vertices for this subpath (flat list [x,y,x,y...])
            sub_indices = []  # Indices relative to the start of this subpath's vertices

            if isinstance(subpath_item, dict) and subpath_item.get('type') == 'roundRect':
                # Generate geometry for roundRect fill (approximated as rectangle)
                params = subpath_item
                x, y, w, h = params['x'], params['y'], params['w'], params['h']
                # TODO: Properly triangulate the roundRect shape for smooth gradients.
                # Simplification: Use vertices of the bounding rectangle.
                # This will result in gradients not curving around corners smoothly.
                sub_vertices = [
                    x, y,         # Top-Left (Vertex 0 relative to subpath)
                    x + w, y,     # Top-Right (Vertex 1)
                    x + w, y + h, # Bottom-Right (Vertex 2)
                    x, y + h      # Bottom-Left (Vertex 3)
                ]
                # Indices for two triangles covering the rectangle (relative indices 0, 1, 2, 3)
                sub_indices = [0, 1, 2,  0, 2, 3] # Triangles: (TL, TR, BR), (TL, BR, BL)
                warnings.warn("Gradient fill on roundRect is approximated using rectangle geometry. Rounded corners will not have smooth gradient.", stacklevel=4) # Deeper stacklevel as it's called by fill()

            elif isinstance(subpath_item, list):
                # Polygon list subpath
                point_tuples = subpath_item
                num_points = len(point_tuples)
                if num_points >= 3: # Need at least 3 points for a fillable polygon
                    # Add vertices (x, y) pairs flattened
                    for px, py in point_tuples:
                        sub_vertices.extend([px, py])

                    # Simple triangle fan triangulation (relative indices: 0, 1, 2...)
                    # WARNING: Assumes polygon is convex and not self-intersecting for correct results.
                    # Indices: (0, 1, 2), (0, 2, 3), ... referencing vertex index within subpath.
                    for i in range(1, num_points - 1):
                        sub_indices.extend([0, i, i + 1])
                # else: Polylines with < 3 points cannot be filled. Skip.

            # --- Accumulate Geometry from this Subpath ---
            if sub_vertices and sub_indices:
                num_sub_verts_added = len(sub_vertices) // 2 # Number of (x,y) pairs added
                # Adjust indices to be absolute (relative to the start of all_vertices)
                # by adding the current vertex_offset to each relative index.
                adjusted_indices = [idx + vertex_offset for idx in sub_indices]

                all_vertices.extend(sub_vertices) # Append flat vertex data
                all_indices.extend(adjusted_indices) # Append adjusted indices
                # Update offset for the next subpath
                vertex_offset += num_sub_verts_added

        # Check if any geometry was generated across all subpaths
        if not all_vertices or not all_indices:
            return None

        # Return dictionary format expected by gradient rendering and potentially fill() stencil
        return {'vertices': all_vertices, 'indices': all_indices}


    def _render_gradient_with_shader(self, gradient: CanvasGradient, geometry: dict):
        """
        Internal: Configures the gradient shader RenderContext and draws a Mesh
        using the provided geometry.
        Assumes called within a Kivy canvas context (`with self.canvas:` or similar).
        Handles setting uniforms and adding/removing RenderContext instruction.
        """
        if not self._render_context:
            Logger.error("Canvas2DContext: Cannot render gradient, shader RenderContext is not available.")
            return
        if not geometry or 'vertices' not in geometry or 'indices' not in geometry:
            warnings.warn("Attempted to render gradient with invalid or missing geometry.", stacklevel=3)
            return
        if not hasattr(gradient, 'color_stops'):
             warnings.warn("Attempted to render gradient with invalid gradient object.", stacklevel=3)
             return

        vertices = geometry.get('vertices') # Flat list [x1,y1, x2,y2,...]
        indices = geometry.get('indices')   # List [i1, i2, i3,...]

        # Validate geometry data received
        # Need at least 3 vertices (6 floats) and 3 indices for one triangle.
        if not vertices or not indices or len(indices) < 3 or len(vertices) < 6:
            warnings.warn(f"Insufficient geometry for gradient mesh: {len(vertices)} floats, {len(indices)} indices", stacklevel=3)
            return

        # Validate and prepare color stops
        num_stops = len(gradient.color_stops)
        if num_stops == 0:
             warnings.warn("Gradient has no color stops, cannot render.", stacklevel=3)
             return # Cannot draw gradient with no stops
        if num_stops > self._MAX_COLOR_STOPS:
             warnings.warn(f"Gradient has {num_stops} stops, exceeding shader limit ({self._MAX_COLOR_STOPS}). Truncating stops.", stacklevel=3)
             num_stops = self._MAX_COLOR_STOPS
             # Use only the first MAX_STOPS (they are already sorted by offset in addColorStop)
             active_stops = gradient.color_stops[:self._MAX_COLOR_STOPS]
        else:
             active_stops = gradient.color_stops

        # --- Prepare Uniforms Dictionary ---
        uniforms = {}
        uniforms['u_global_alpha'] = float(self._globalAlpha) # Pass global alpha
        uniforms['u_num_color_stops'] = int(num_stops)

        # Create flat arrays padded to MAX_STOPS for offsets and colors
        # Offsets: [o1, o2, ..., oN, 0.0, ..., 0.0]
        offsets_padded = [0.0] * self._MAX_COLOR_STOPS
        # Colors: [r1,g1,b1,a1, r2,g2,b2,a2, ..., rN,gN,bN,aN, 0,0,0,0, ..., 0,0,0,0]
        colors_padded = [0.0] * (self._MAX_COLOR_STOPS * 4)

        for i in range(num_stops):
            # Ensure stop data is valid (should be if addColorStop worked)
            if len(active_stops[i]) == 2 and len(active_stops[i][1]) == 4:
                offset, (r, g, b, a) = active_stops[i]
                offsets_padded[i] = float(offset)
                color_base_index = i * 4
                colors_padded[color_base_index + 0] = float(r)
                colors_padded[color_base_index + 1] = float(g)
                colors_padded[color_base_index + 2] = float(b)
                colors_padded[color_base_index + 3] = float(a)
            else:
                 warnings.warn(f"Invalid color stop format found at index {i}. Skipping stop.", stacklevel=4)
                 # Handle potentially bad data gracefully - maybe reduce num_stops?
                 # For now, it will just have zeros padded.

        # Assign padded lists/tuples directly to uniform names
        # Kivy handles converting these Python lists into appropriate GL uniform arrays.
        uniforms['u_offsets'] = offsets_padded
        uniforms['u_colors'] = colors_padded

        # Gradient Type Specific Uniforms
        gradient_type_int = -1 # Default/invalid type
        try:
            if isinstance(gradient, LinearGradient):
                gradient_type_int = 0
                uniforms['u_linear_start'] = (float(gradient.x0), float(gradient.y0))
                uniforms['u_linear_end'] = (float(gradient.x1), float(gradient.y1))
            elif isinstance(gradient, RadialGradient):
                gradient_type_int = 1
                uniforms['u_radial_c0'] = (float(gradient.x0), float(gradient.y0))
                uniforms['u_radial_r0'] = float(gradient.r0)
                uniforms['u_radial_c1'] = (float(gradient.x1), float(gradient.y1))
                uniforms['u_radial_r1'] = float(gradient.r1)
            elif isinstance(gradient, ConicGradient):
                gradient_type_int = 2
                uniforms['u_conic_angle'] = float(gradient.start_angle) # Radians
                uniforms['u_conic_center'] = (float(gradient.cx), float(gradient.cy))
            else:
                # Should not happen if type checked earlier, but safeguard.
                warnings.warn(f"Unsupported gradient type for shader: {type(gradient)}", stacklevel=3)
                return # Cannot render unknown type

            uniforms['u_gradient_type'] = gradient_type_int

        except (ValueError, TypeError) as e:
            warnings.warn(f"Invalid coordinate or parameter in gradient object: {e}", stacklevel=3)
            return # Cannot render if gradient data is invalid

        # --- Setup RenderContext and Draw Mesh ---
        rc = self._render_context # Get the shared RenderContext

        # Set all collected uniforms on the RenderContext
        # Use dictionary assignment which Kivy's RenderContext supports.
        for name, value in uniforms.items():
            rc[name] = value
        # Alternative: set individually: rc['u_name'] = value

        # Create the Mesh instruction using the prepared geometry and shader format
        # Need x, y only for vPosition
        mesh_vertices_pos_only = []
        for i in range(0, len(vertices), 2): # Iterate taking x,y pairs
             mesh_vertices_pos_only.extend([vertices[i], vertices[i+1]])

        gradient_mesh = Mesh(
            vertices=mesh_vertices_pos_only, # Flat list [x1,y1, x2,y2,...] in Canvas coords
            indices=indices,   # List of vertex indices [i1, i2, i3,...]
            fmt=self._gradient_vertex_format_shader, # Use shader format: [(b'vPosition', 2, 'float')]
            mode='triangles'   # Draw triangles
            # Texture is not used for gradient mesh
        )

        # Add RenderContext and Mesh to the current canvas instructions
        # This must happen within the correct Kivy drawing context (e.g., `with self.canvas:`).
        self.canvas.add(rc) # Add RenderContext to activate shader for the next instruction(s)
        self.canvas.add(gradient_mesh) # Add Mesh to be drawn using the active shader

        # CRITICAL: Remove the RenderContext instruction immediately *after* this mesh
        # is conceptually drawn. Use a Kivy Callback instruction to schedule the removal.
        # This ensures the shader doesn't affect subsequent drawing.
        # Need to capture 'rc' safely in the callback lambda.
        callback = Callback(lambda *args, _rc=rc: self.canvas.remove(_rc) if _rc in self.canvas.children else None,
                           at_least_once=True) # Ensure callback runs even if frame skips? Test necessity.
        self.canvas.add(callback)


    # --- Canvas API Methods ---

    # --- State Management ---
    def save(self) -> None:
        """Saves the current drawing state onto a stack."""
        # Deep copy mutable states: matrices, lists (lineDash), Path2D objects
        # Shallow copy immutable states: numbers, strings, tuples (colors)
        # Gradient objects are treated as immutable after creation (shallow copy).

        fill_style_copy = self._fillStyle # Shallow copy is fine for tuple or gradient obj
        stroke_style_copy = self._strokeStyle # Shallow copy

        state = {
            # Styles
            'fillStyle': fill_style_copy,
            'strokeStyle': stroke_style_copy,
            # Line Styles
            'lineWidth': self._lineWidth,
            'lineCap': self._lineCap,
            'lineJoin': self._lineJoin,
            'miterLimit': self._miterLimit,
            'lineDash': self._lineDash[:], # Explicit copy of list
            'lineDashOffset': self._lineDashOffset,
            # Compositing & Filters
            'globalAlpha': self._globalAlpha,
            'globalCompositeOperation': self._globalCompositeOperation,
            'imageSmoothingEnabled': self._imageSmoothingEnabled,
            'imageSmoothingQuality': self._imageSmoothingQuality,
            'filter': self._filter,
            # Text Styles
            'font': self._font, # String is immutable
            'textAlign': self._textAlign,
            'textBaseline': self._textBaseline,
            'direction': self._direction,
            # Shadow Styles (tuples are immutable)
            'shadowBlur': self._shadowBlur,
            'shadowColor': self._shadowColor,
            'shadowOffsetX': self._shadowOffsetX,
            'shadowOffsetY': self._shadowOffsetY,
            # Transform Matrix
            'user_matrix': self._user_matrix.get(), # Explicit copy of Matrix object
            # Clipping State (Path2D needs deep copy)
            'clip_path': copy.deepcopy(self._clip_path) if self._clip_path else None,
            'clip_fill_rule': self._clip_fill_rule,
            # TODO: Save/Restore current path? Spec is ambiguous. Let's assume beginPath clears it.
            # 'current_path': copy.deepcopy(self._current_path),
        }
        self._state_stack.append(state)

    def restore(self) -> None:
        """Restores the most recently saved drawing state from the stack."""
        if not self._state_stack:
            # Spec: If stack is empty, do nothing.
            return
        state = self._state_stack.pop()

        # Restore state variables directly from the saved dictionary
        self._fillStyle = state['fillStyle']
        self._strokeStyle = state['strokeStyle']
        self._lineWidth = state['lineWidth']
        self._lineCap = state['lineCap']
        self._lineJoin = state['lineJoin']
        self._miterLimit = state['miterLimit']
        self._lineDash = state['lineDash'] # Already a copy from save
        self._lineDashOffset = state['lineDashOffset']
        self._globalAlpha = state['globalAlpha']
        self._globalCompositeOperation = state['globalCompositeOperation']
        self._imageSmoothingEnabled = state['imageSmoothingEnabled']
        self._imageSmoothingQuality = state['imageSmoothingQuality']
        self._filter = state['filter']

        # Restore font using the setter to update derived properties (_font_size, etc.)
        # This re-parses the font string.
        self.font = state['font']
        # Ensure other text properties are restored AFTER font potentially changed them
        self._textAlign = state['textAlign']
        self._textBaseline = state['textBaseline']
        self._direction = state['direction']

        self._shadowBlur = state['shadowBlur']
        self._shadowColor = state['shadowColor']
        self._shadowOffsetX = state['shadowOffsetX']
        self._shadowOffsetY = state['shadowOffsetY']

        # Restore transform matrix (was copied in save)
        self._user_matrix.set(flat=state['user_matrix'])

        # Restore clipping state (was deep copied in save)
        self._clip_path = state['clip_path']
        self._clip_fill_rule = state['clip_fill_rule']

        # TODO: Restore current path if it was saved?
        # self._current_path = state['current_path']

    def reset(self) -> None:
        """Resets the context state, transformation, clipping, and clears the canvas."""
        # Reset all state variables to their default values
        self._fillStyle = self.DEFAULT_FILL_STYLE
        self._strokeStyle = self.DEFAULT_STROKE_STYLE
        self._lineWidth = self.DEFAULT_LINE_WIDTH
        self._lineCap = self.DEFAULT_LINE_CAP
        self._lineJoin = self.DEFAULT_LINE_JOIN
        self._miterLimit = self.DEFAULT_MITER_LIMIT
        self._lineDash = []
        self._lineDashOffset = 0.0
        self._globalAlpha = self.DEFAULT_GLOBAL_ALPHA
        self._globalCompositeOperation = self.DEFAULT_COMPOSITE_OP
        self._imageSmoothingEnabled = self.DEFAULT_IMAGE_SMOOTHING
        self._imageSmoothingQuality = self.DEFAULT_IMAGE_SMOOTHING_QUALITY
        self._filter = self.DEFAULT_FILTER
        # Reset font using setter with default string to ensure derived props are correct
        self.font = self.DEFAULT_FONT
        self._textAlign = self.DEFAULT_TEXT_ALIGN
        self._textBaseline = self.DEFAULT_TEXT_BASELINE
        self._direction = self.DEFAULT_DIRECTION
        self._shadowBlur = self.DEFAULT_SHADOW_BLUR
        self._shadowColor = self.DEFAULT_SHADOW_COLOR
        self._shadowOffsetX = self.DEFAULT_SHADOW_OFFSET_X
        self._shadowOffsetY = self.DEFAULT_SHADOW_OFFSET_Y

        # Reset transform matrix to identity
        self.resetTransform()

        # Reset clipping path state
        self._clip_path = None
        self._clip_fill_rule = 'nonzero'

        # Reset current path object being built
        self._current_path = Path2D()

        # Clear the state stack
        self._state_stack = []

        # Reset the underlying Kivy canvas instructions
        self._setup_canvas()
        # Ensure base matrix is correct for current size/pos (might have changed since init)
        self._update_base_matrix()
        # --- *** 修改点 2: 移除这里的直接更新 *** ---
        # REMOVED: self._update_derived_font_properties()
        # The initial update is scheduled in __init__. Subsequent resets will update via the font setter.

        # *** 确保背景矩形在 reset 后尺寸正确 ***
        self._update_background()


    # --- Transformations (affect _user_matrix) ---
    def scale(self, x: float, y: float | None = None) -> None:
        """Applies scaling transformation to the user matrix."""
        try:
            sx = float(x)
            # Spec: If y is omitted, it defaults to x
            sy = sx if y is None else float(y)
            # Spec: If either argument is infinite or NaN, ignore the call.
            if not (math.isfinite(sx) and math.isfinite(sy)):
                warnings.warn(f"Ignoring scale() call with non-finite values: ({x}, {y}).", stacklevel=2)
                return
            # Kivy's matrix.scale() multiplies the matrix in place.
            self._user_matrix.scale(sx, sy, 1.0) # Scale Z by 1 (unused in 2D)
        except (ValueError, TypeError) as e:
             warnings.warn(f"Invalid scale values: ({x}, {y}) -> {e}. Ignoring call.", stacklevel=2)

    def rotate(self, angle: float) -> None:
        """Applies rotation transformation (angle in radians, clockwise)."""
        try:
            a_rad = float(angle)
            if not math.isfinite(a_rad):
                warnings.warn(f"Ignoring rotate() call with non-finite angle: {angle}.", stacklevel=2)
                return
            self._user_matrix.rotate(angle = a_rad, x = 0, y = 0, z = 1) # Rotate around Z axis
        except (ValueError, TypeError) as e:
             warnings.warn(f"Invalid rotation angle: {angle} -> {e}. Ignoring call.", stacklevel=2)

    def translate(self, x: float, y: float) -> None:
        """Applies translation transformation."""
        try:
            tx, ty = float(x), float(y)
            # Spec: If either argument is infinite or NaN, ignore the call.
            if not (math.isfinite(tx) and math.isfinite(ty)):
                warnings.warn(f"Ignoring translate() call with non-finite values: ({x}, {y}).", stacklevel=2)
                return
            # Kivy's matrix.translate() modifies in place.
            self._user_matrix.translate(tx, ty, 0) # Translate Z by 0
        except (ValueError, TypeError) as e:
             warnings.warn(f"Invalid translate values: ({x}, {y}) -> {e}. Ignoring call.", stacklevel=2)

    def transform(self, a: float, b: float, c: float, d: float, e: float, f: float) -> None:
        """Multiplies the current user matrix by the given 2D transform matrix."""
        try:
            # Convert all to float first
            vals = [float(v) for v in (a, b, c, d, e, f)]
            # Spec: If any argument is infinite or NaN, ignore the call.
            if not all(math.isfinite(v) for v in vals):
                warnings.warn(f"Ignoring transform() call with non-finite values.", stacklevel=2)
                return

            # Create a Kivy matrix representing the input 2D transform (a,b,c,d,e,f)
            # Kivy Matrix layout (column-major, stored flat):
            # [ a  c  e ]   [ m[0] m[4] m[8]  m[12] ] -> [ a c 0 e ] (col 0, 1, 2, 3)
            # [ b  d  f ]   [ m[1] m[5] m[9]  m[13] ] -> [ b d 0 f ]
            # [ 0  0  1 ]   [ m[2] m[6] m[10] m[14] ] -> [ 0 0 1 0 ]
            #               [ m[3] m[7] m[11] m[15] ] -> [ 0 0 0 1 ]
            mat_transform = Matrix()
            mat_transform.set(flat=[
                vals[0], vals[1], 0, 0,  # Column 0 (a, b, 0, 0)
                vals[2], vals[3], 0, 0,  # Column 1 (c, d, 0, 0)
                0,       0,       1, 0,  # Column 2 (0, 0, 1, 0) - Z axis identity
                vals[4], vals[5], 0, 1   # Column 3 (e, f, 0, 1) - Translation
            ])

            self._user_matrix = self._user_matrix.multiply(mat_transform)
        except (ValueError, TypeError) as e:
             warnings.warn(f"Invalid transform values -> {e}. Ignoring call.", stacklevel=2)

    def setTransform(self, a: float, b: float, c: float, d: float, e: float, f: float) -> None:
        """Resets the user transform to identity, then applies the given matrix."""
        try:
            # Convert all to float first
            vals = [float(v) for v in (a, b, c, d, e, f)]
            # Spec: If any argument is infinite or NaN, reset to identity and ignore args.
            if not all(math.isfinite(v) for v in vals):
                 warnings.warn(f"Non-finite values in setTransform. Resetting transform to identity.", stacklevel=2)
                 self.resetTransform() # Reset to identity
                 return

            # If args are valid, set the user matrix directly using the values
            self._user_matrix.set(array=[
                [vals[0], vals[1], 0.0, 0.0],  # 第一列 (a, b, 0, 0)
                [vals[2], vals[3], 0.0, 0.0],  # 第二列 (c, d, 0, 0)
                [0.0, 0.0, 1.0, 0.0],         # 第三列 (0, 0, 1, 0)
                [vals[4], vals[5], 0.0, 1.0]  # 第四列 (e, f, 0, 1)
            ])
        except (ValueError, TypeError) as e:
             # Should not happen if float conversion works, but catch just in case
             warnings.warn(f"Invalid setTransform values -> {e}. Resetting transform.", stacklevel=2)
             self.resetTransform()

    def resetTransform(self) -> None:
        """Resets the user transformation matrix to the identity matrix."""
        self._user_matrix.identity()

    def getTransform(self) -> Matrix:
        """Returns a copy of the current user transformation matrix (Kivy Matrix)."""
        # Note: Canvas API spec requires returning a DOMMatrix object.
        # Returning Kivy's Matrix is the practical equivalent in this implementation.
        # The user needs to know how to interpret Kivy's Matrix structure if needed.
        return self._user_matrix.copy()


    # --- Rectangles ---
    def clearRect(self, x: float, y: float, w: float, h: float) -> None:
        """Clears the specified rectangular area, making it transparent black."""
        # Spec: clearRect ignores shadows, globalAlpha, globalCompositeOperation.
        # It IS affected by transform and clipping.
        try:
            x_f, y_f, w_f, h_f = map(float, [x, y, w, h])
            # Spec: If any arg is non-finite, ignore. If w/h is 0, ignore.
            if not all(math.isfinite(v) for v in (x_f, y_f, w_f, h_f)) or w_f == 0 or h_f == 0:
                 return

            # Handle negative width/height by adjusting start point and using abs size
            draw_x = x_f if w_f >= 0 else x_f + w_f
            draw_y = y_f if h_f >= 0 else y_f + h_f
            draw_w = abs(w_f)
            draw_h = abs(h_f)
            if draw_w <= 0 or draw_h <= 0: return # Ignore zero area

            # Implement by drawing a rectangle with Color(0,0,0,0) inside the current transform/clip state.
            # This works visually if background is opaque, but doesn't truly "clear" underlying pixels
            # if the canvas itself has transparency or content 'behind' it.
            # True clearing might need specific blend funcs (GL_ZERO, GL_ZERO) or FBO manipulation.
            # Let's stick to the simpler visual clearing for now.

            self._push_drawing_state() # Apply transform & clip
            with self.canvas:
                Color(1, 1, 1, 1)
                Rectangle(pos=(draw_x, draw_y), size=(draw_w, draw_h))
            self._pop_drawing_state() # Restore transform/clip

        except (ValueError, TypeError):
             # Ignore calls with invalid numeric types
             warnings.warn(f"Invalid clearRect values: {(x, y, w, h)}. Ignoring call.", stacklevel=2)

    def fillRect(self, x: float, y: float, w: float, h: float) -> None:
        """Fills the given rectangle with the current fillStyle."""
        try:
            x_f, y_f, w_f, h_f = map(float, [x, y, w, h])
            # Ignore if non-finite or zero size
            if not all(math.isfinite(v) for v in (x_f, y_f, w_f, h_f)) or w_f == 0 or h_f == 0:
                 return
            draw_x = x_f if w_f >= 0 else x_f + w_f
            draw_y = y_f if h_f >= 0 else y_f + h_f
            draw_w = abs(w_f)
            draw_h = abs(h_f)
            if draw_w <= 0 or draw_h <= 0: return

            style = self._fillStyle # Get current fill style

            self._push_drawing_state() # Apply transform & clip
            with self.canvas: # Instructions added here are affected by pushed state
                if hasattr(style, '_is_kivy_gradient') and style._is_kivy_gradient:
                    # --- Render Gradient using Shader ---
                    # Define geometry for the rectangle in Canvas coordinates
                    # Vertices: TL, TR, BR, BL (order matters for indices)
                    # Need only x,y for vPosition format
                    rect_geom = {
                        'vertices': [
                            draw_x, draw_y,                 # Top-Left (Vertex 0)
                            draw_x + draw_w, draw_y,         # Top-Right (Vertex 1)
                            draw_x + draw_w, draw_y + draw_h, # Bottom-Right (Vertex 2)
                            draw_x, draw_y + draw_h          # Bottom-Left (Vertex 3)
                        ],
                        # Indices for two triangles covering the rectangle: (0,1,2), (0,2,3)
                        'indices': [0, 1, 2,  0, 2, 3]
                    }
                    # Call the helper to setup shader uniforms and draw the mesh
                    self._render_gradient_with_shader(style, rect_geom)
                    # RenderContext removal is handled by callback in the helper

                else: # Try solid color or pattern (pattern not implemented)
                     color_rgba = self._get_current_color('fill') # Applies globalAlpha
                     if color_rgba and color_rgba[3] > 0: # Check if color is valid and visible
                         Color(*color_rgba) # Set Kivy color instruction
                         # Kivy Rectangle pos/size are interpreted relative to current transform
                         Rectangle(pos=(draw_x, draw_y), size=(draw_w, draw_h))
                     # else: Style was pattern or invalid/transparent color, draw nothing

            self._pop_drawing_state() # Restore transform/clip

        except (ValueError, TypeError):
             warnings.warn(f"Invalid fillRect values: {(x, y, w, h)}. Ignoring call.", stacklevel=2)


    def strokeRect(self, x: float, y: float, w: float, h: float) -> None:
        """Strokes the outline of the given rectangle with the current strokeStyle."""
        try:
            x_f, y_f, w_f, h_f = map(float, [x, y, w, h])
            # Ignore if non-finite, zero size, or zero line width
            if not all(math.isfinite(v) for v in (x_f, y_f, w_f, h_f)) or w_f == 0 or h_f == 0 or self._lineWidth <= 0:
                return
            draw_x = x_f if w_f >= 0 else x_f + w_f
            draw_y = y_f if h_f >= 0 else y_f + h_f
            draw_w = abs(w_f)
            draw_h = abs(h_f)
            if draw_w <= 0 or draw_h <= 0: return

            # Get stroke color (handles gradient fallback and globalAlpha)
            stroke_color_rgba = self._get_current_color('stroke')

            # Check if stroke style is valid and visible
            if not stroke_color_rgba or stroke_color_rgba[3] == 0:
                return # Nothing to stroke if color is invalid or transparent

            # --- FIX START: Map lineCap to Kivy supported values ---
            kivy_cap = 'none' # Default mapping for 'butt' and invalid values
            if self._lineCap == 'round':
                kivy_cap = 'round'
            elif self._lineCap == 'square':
                kivy_cap = 'square'
            # --- FIX END ---

            self._push_drawing_state() # Apply transform & clip
            with self.canvas:
                Color(*stroke_color_rgba) # Set the stroke color
                line_inst = Line(rectangle=(draw_x, draw_y, draw_w, draw_h),
                                  width=self._lineWidth,
                                  joint=self._lineJoin,
                                  cap=kivy_cap)
                 # Apply miter limit if join is 'miter'
                 # if self._lineJoin == 'miter':
                 #     # AttributeError: 'Line' object has no attribute 'miter_limit'
                 #     # Kivy's Line might not support setting this directly after creation,
                 #     # or for the 'rectangle' mode. Using Kivy's default limit.
                 #     # line_inst.miter_limit = self._miterLimit  # REMOVED/COMMENTED OUT
                 # Instruction added automatically in `with` block

            self._pop_drawing_state() # Restore transform/clip
        except (ValueError, TypeError):
             warnings.warn(f"Invalid strokeRect values: {(x, y, w, h)}. Ignoring call.", stacklevel=2)


    # --- Text ---
    def _get_text_draw_pos(self, text: str, x: float, y: float, metrics: TextMetrics) -> tuple[float, float]:
        """Calculates the Kivy drawing position (bottom-left of text bounding box in *Canvas* coordinates)
           based on Canvas x, y, textAlign, and textBaseline.
           The final screen position is determined by the transform matrix."""

        text_w = metrics.width # Measured width of the text

        # --- Horizontal Alignment (Calculate offset from Canvas 'x') ---
        align = self._textAlign
        direction = self._direction # 'ltr' or 'rtl'

        # Resolve 'start' and 'end' based on direction
        effective_align = align
        if align == 'start':
            effective_align = 'left' if direction == 'ltr' else 'right'
        elif align == 'end':
            effective_align = 'right' if direction == 'ltr' else 'left'

        # Calculate the Canvas X coordinate for the *left* edge of the text bounding box
        canvas_left_x = x # Default for align='left'
        if effective_align == 'center':
            canvas_left_x = x - text_w / 2.0
        elif effective_align == 'right':
            canvas_left_x = x - text_w

        baseline = self._textBaseline
        canvas_bottom_y = y
        if baseline == 'alphabetic': canvas_bottom_y = y - metrics.alphabeticBaseline
        elif baseline == 'top': canvas_bottom_y = y
        elif baseline == 'hanging': canvas_bottom_y = y - metrics.hangingBaseline
        elif baseline == 'middle': canvas_bottom_y = y - metrics.font_height / 2
        elif baseline == 'ideographic': canvas_bottom_y = y - metrics.ideographicBaseline
        elif baseline == 'bottom': canvas_bottom_y = y - metrics.font_height

        return canvas_left_x, canvas_bottom_y

    def fillText(self, text: str, x: float, y: float, maxWidth: float | None = None) -> None:
        """Draws filled text using the current fillStyle."""
        text_str = str(text)
        if not text_str: return # Don't draw empty strings

        # Get fill color (handles gradient fallback and globalAlpha)
        text_rgba = self._get_current_color('fill')
        if not text_rgba or text_rgba[3] == 0: return # No valid/visible color

        try:
            # Validate coordinates and maxWidth
            x_f, y_f = float(x), float(y)
            mw_f = float(maxWidth) if maxWidth is not None else None
            if not (math.isfinite(x_f) and math.isfinite(y_f)):
                 warnings.warn(f"Non-finite coordinates in fillText: ({x}, {y}).", stacklevel=2)
                 return
            if mw_f is not None and not math.isfinite(mw_f): mw_f = None
            if mw_f is not None and mw_f <= 0: return

            # --- Text Measurement & Preparation ---
            metrics = TextMetrics(text_str, self)
            base_w = metrics.width
            base_h = metrics._text_height
            if base_w <= 0 or base_h <= 0: return

            label = metrics._label
            label.options['color'] = text_rgba
            label.refresh()
            texture = label.texture
            if not texture:
                 warnings.warn(f"Failed to render texture for fillText: '{text_str}'", stacklevel=3)
                 return
            tex_w, tex_h = texture.size # Get texture dimensions AFTER refresh

            # --- Handle maxWidth ---
            draw_w = base_w
            if mw_f is not None and base_w > mw_f:
                scale_x = mw_f / base_w
                draw_w = mw_f
                warnings.warn("fillText maxWidth constraint applied by scaling text horizontally, which may distort appearance.", stacklevel=3)
            draw_h = base_h

            # --- Calculate Drawing Position ---
            canvas_bl_x, canvas_bl_y = self._get_text_draw_pos(text_str, x_f, y_f, metrics)

            # --- FIX: Use UV calculation logic derived from drawImage ---
            # Source rectangle is the whole texture: sx=0, sy=0, sw=tex_w, sh=tex_h
            # Calculate Kivy texture coordinates (u, v) with BL origin [0, 1]
            # based on the Canvas source rect (TL origin).
            sx, sy, sw, sh = 0.0, 0.0, float(tex_w), float(tex_h)
            u0 = sx / tex_w
            u1 = (sx + sw) / tex_w
            # Calculate V for the top (sy) and bottom (sy + sh) edges in Canvas space
            v_top_canvas = sy / tex_h
            v_bottom_canvas = (sy + sh) / tex_h
            # Convert to Kivy V coords (Bottom-Left origin)
            v0_kivy = 1.0 - v_bottom_canvas # Kivy V for bottom edge (maps to Canvas Y = sh) -> 1 - (sh/sh) = 0
            v1_kivy = 1.0 - v_top_canvas    # Kivy V for top edge (maps to Canvas Y = 0) -> 1 - 0 = 1

            # Kivy Rectangle `tex_coords` order: BL, BR, TR, TL (u,v pairs flattened)
            # This results in the default texture coordinates [0, 0, 1, 0, 1, 1, 0, 1]
            # when sx=0, sy=0, sw=tex_w, sh=tex_h
            calculated_tex_coords = [
                u0, v0_kivy,  # BL vertex -> Tex BL (0,0)
                u1, v0_kivy,  # BR vertex -> Tex BR (1,0)
                u1, v1_kivy,  # TR vertex -> Tex TR (1,1)
                u0, v1_kivy   # TL vertex -> Tex TL (0,1)
            ]
            # --- END FIX ---

            # --- Draw Text ---
            self._push_drawing_state() # Apply transform & clip
            with self.canvas:
                 # Color is baked into the texture, so use white here. Alpha handled by texture.
                 Color(1, 1, 1, 1)
                 # Set texture filtering based on image smoothing settings
                 filter_mode = self._get_smoothing_filter()
                 texture.mag_filter = filter_mode
                 texture.min_filter = filter_mode

                 # Draw using the calculated position, size, texture, and FIXED tex_coords
                 Rectangle(pos=(canvas_bl_x, canvas_bl_y), # Position is bottom-left in Canvas Coords
                           size=(draw_w, draw_h),         # Use potentially scaled width
                           texture=texture,
                           tex_coords=calculated_tex_coords) # <--- APPLY THE FIX HERE

            self._pop_drawing_state() # Restore transform/clip

        except (ValueError, TypeError) as e:
             warnings.warn(f"Invalid fillText arguments or measurement failed: {(text, x, y, maxWidth)} -> {e}.", stacklevel=2)


    # ================================================================
    # --- REVISED strokeText Implementation (Two-Pass Technique) ---
    # ================================================================
    def strokeText(self, text: str, x: float, y: float, maxWidth: float | None = None) -> None:
        """
        Draws the outline of text using a two-pass technique to simulate
        Web Canvas strokeText (outline only).

        Requires a matching opaque background color set in canvas.before.
        """
        text_str = str(text)
        stroke_rgba = self._get_current_color('stroke') # Includes globalAlpha

        # --- Basic Validation ---
        if not text_str or self._lineWidth <= 0 or not stroke_rgba or stroke_rgba[3] == 0:
            return

        # --- Define Erase Color (MUST MATCH ACTUAL BACKGROUND) ---
        # Assuming the background set in __init__ is opaque white
        erase_color = (1, 1, 1, 1) # Opaque White

        try:
            # --- Validate Coords & MaxWidth ---
            x_f, y_f = float(x), float(y)
            mw_f = float(maxWidth) if maxWidth is not None else None
            if not (math.isfinite(x_f) and math.isfinite(y_f)):
                 warnings.warn(f"Non-finite coordinates in strokeText: ({x}, {y}).", stacklevel=2); return
            if mw_f is not None and not math.isfinite(mw_f): mw_f = None
            if mw_f is not None and mw_f <= 0: return

            # --- Measurement (using standard fill metrics for alignment) ---
            metrics = TextMetrics(text_str, self)
            base_w = metrics.width
            if base_w <= 0: return

            # --- Pass 1: Render Outline Layer (Filled + Outlined with strokeStyle) ---
            kivy_outline_width = max(0.0, self._lineWidth / 2.0)
            outline_label = CoreLabel(
                text=text_str,
                font_size=self._font_size, font_name=self._font_name,
                bold=self._font_bold, italic=self._font_italic,
                color=stroke_rgba,          # Fill with stroke color
                outline_color=stroke_rgba,  # Outline with stroke color
                outline_width=kivy_outline_width
            )
            outline_label.refresh()
            texture_outline = outline_label.texture
            if not texture_outline:
                 warnings.warn(f"Failed P1 texture render strokeText: '{text_str}'", stacklevel=3); return
            tex_w_outline, tex_h_outline = texture_outline.size
            if tex_w_outline <= 0 or tex_h_outline <= 0: return

            # --- Pass 2: Render Fill Layer (Filled with Background Color, No Outline) ---
            fill_label = CoreLabel(
                text=text_str,
                font_size=self._font_size, font_name=self._font_name,
                bold=self._font_bold, italic=self._font_italic,
                color=erase_color, # Fill with background color
                outline_width=0    # No outline
            )
            fill_label.refresh()
            texture_fill = fill_label.texture
            if not texture_fill:
                 warnings.warn(f"Failed P2 texture render strokeText: '{text_str}'", stacklevel=3); return
            tex_w_fill, tex_h_fill = texture_fill.size
            if tex_w_fill <= 0 or tex_h_fill <= 0: return # Should be same size as metrics base_w/h

            # --- Handle maxWidth (Validation only - No scaling applied) ---
            # Ignore maxWidth for rendering stroke visually

            # --- Calculate Drawing Positions ---
            # Base position calculated from fill metrics
            canvas_bl_x, canvas_bl_y = self._get_text_draw_pos(text_str, x_f, y_f, metrics)

            # Offset for the outline layer texture (it's larger)
            # Position the outline layer slightly down/left so the fill layer aligns centrally
            outline_offset_x = (tex_w_outline - tex_w_fill) / 2.0
            outline_offset_y = (tex_h_outline - tex_h_fill) / 2.0

            pos_outline = (canvas_bl_x - outline_offset_x, canvas_bl_y - outline_offset_y)
            pos_fill = (canvas_bl_x, canvas_bl_y) # Fill layer aligns directly

            # --- Calculate UVs (Full texture for both) ---
            uvs = [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0] # Standard [0,0, 1,0, 1,1, 0,1] mapping

            # --- Draw Stroked Text (Two Passes) ---
            self._push_drawing_state()
            with self.canvas:
                 # Set filtering (apply to both)
                 filter_mode = self._get_smoothing_filter()
                 texture_outline.mag_filter = filter_mode; texture_outline.min_filter = filter_mode
                 texture_fill.mag_filter = filter_mode; texture_fill.min_filter = filter_mode

                 # Pass 1: Draw the larger outline layer (already has stroke color baked in)
                 Color(1, 1, 1, 1) # Use full opacity, color is in texture
                 Rectangle(pos=pos_outline, size=(tex_w_outline, tex_h_outline),
                           texture=texture_outline, tex_coords=uvs)

                 # Pass 2: Draw the smaller fill layer (background color) on top to "erase" center
                 Color(1, 1, 1, 1) # Use full opacity, color is in texture
                 Rectangle(pos=pos_fill, size=(tex_w_fill, tex_h_fill),
                           texture=texture_fill, tex_coords=uvs)

            self._pop_drawing_state()

        except (ValueError, TypeError) as e:
             warnings.warn(f"Invalid strokeText args or rendering failed: {e}.", stacklevel=2)
        except Exception as e:
             Logger.error(f"Unexpected error in strokeText (2-pass): {e}", exc_info=True)
             warnings.warn(f"Unexpected error processing strokeText (2-pass): {e}", stacklevel=2)
    # ================================================================
    # --- END REVISED strokeText (Two-Pass Technique) ---
    # ================================================================


    def measureText(self, text: str) -> TextMetrics:
        """Measures the given text based on the current font settings."""
        # Ensure font properties derived from the string are up-to-date.
        # This happens automatically in the self.font setter.
        try:
             # Use helper class to perform measurement using Kivy CoreLabel
             return TextMetrics(str(text), self)
        except Exception as e:
             # Handle potential errors during CoreLabel creation/measurement
             warnings.warn(f"measureText failed for '{text}': {e}", stacklevel=2)
             # Return a dummy metrics object with zero values on failure
             class DummyMetrics:
                 width = 0; actualBoundingBoxAscent = 0; actualBoundingBoxDescent = 0
                 actualBoundingBoxLeft = 0; actualBoundingBoxRight = 0
                 fontBoundingBoxAscent = 0; fontBoundingBoxDescent = 0
                 ascent = 0; descent = 0
                 _text_height = 0; _label = None # Add internal fields too?
             return DummyMetrics()


    # --- Paths API (Manipulate self._current_path) ---
    def beginPath(self) -> None:
        """Starts a new path, discarding the current list of subpaths and path state."""
        self._current_path = Path2D()

    def closePath(self) -> None:
        """Attempts to connect the last point of the current subpath back to the first."""
        # Delegates directly to Path2D object
        self._current_path.closePath()

    def moveTo(self, x: float, y: float) -> None:
        """Starts a new subpath at the given (x, y) coordinates."""
        # Delegate to Path2D, which handles validation (finite) and state
        try: self._current_path.moveTo(x, y)
        except (ValueError, TypeError) as e:
             warnings.warn(f"Invalid moveTo coordinates ({x},{y}): {e}", stacklevel=2)

    def lineTo(self, x: float, y: float) -> None:
        """Adds a straight line from the current point to the given (x, y) coordinates."""
        try: self._current_path.lineTo(x, y)
        except (ValueError, TypeError) as e:
             warnings.warn(f"Invalid lineTo coordinates ({x},{y}): {e}", stacklevel=2)

    # --- Path Commands (Delegate to Path2D object, handle/propagate exceptions) ---
    # Wrap Path2D calls in try/except to catch validation errors (non-finite, neg radius)
    # and propagate them as TypeErrors or ValueErrors consistent with spec (simulated).

    def arc(self, x: float, y: float, radius: float, startAngle: float, endAngle: float, counterclockwise: bool = False) -> None:
        """Adds a circular arc to the path."""
        try:
             # Path2D.arc handles validation internally now
             self._current_path.arc(x, y, radius, startAngle, endAngle, bool(counterclockwise))
        except (ValueError, TypeError) as e:
             # Re-raise the specific error (TypeError for non-finite, ValueError for neg radius)
             raise e

    def arcTo(self, x1: float, y1: float, x2: float, y2: float, radius: float) -> None:
        """Adds an arc with the given control points and radius."""
        try:
             self._current_path.arcTo(x1, y1, x2, y2, radius)
        except (ValueError, TypeError) as e: raise e

    def ellipse(self, x: float, y: float, radiusX: float, radiusY: float, rotation: float, startAngle: float, endAngle: float, counterclockwise: bool = False) -> None:
        """Adds an elliptical arc to the path."""
        try:
            self._current_path.ellipse(x, y, radiusX, radiusY, rotation, startAngle, endAngle, bool(counterclockwise))
        except (ValueError, TypeError) as e: raise e

    def bezierCurveTo(self, cp1x: float, cp1y: float, cp2x: float, cp2y: float, x: float, y: float) -> None:
        """Adds a cubic Bézier curve to the path."""
        try:
            # Basic check for finite values before passing to Path2D
            if not all(math.isfinite(v) for v in [cp1x, cp1y, cp2x, cp2y, x, y]):
                 # Spec: Throws TypeError if non-finite
                 raise TypeError("Non-finite coordinate in bezierCurveTo.")
            self._current_path.bezierCurveTo(cp1x, cp1y, cp2x, cp2y, x, y)
        except (ValueError, TypeError) as e: raise e # Let Path2D warnings/errors propagate if needed

    def quadraticCurveTo(self, cpx: float, cpy: float, x: float, y: float) -> None:
        """Adds a quadratic Bézier curve to the path."""
        try:
             if not all(math.isfinite(v) for v in [cpx, cpy, x, y]):
                 raise TypeError("Non-finite coordinate in quadraticCurveTo.")
             self._current_path.quadraticCurveTo(cpx, cpy, x, y)
        except (ValueError, TypeError) as e: raise e

    def rect(self, x: float, y: float, w: float, h: float) -> None:
        """Adds a rectangle subpath."""
        try:
            # Path2D.rect handles validation now
            self._current_path.rect(x, y, w, h)
        except (ValueError, TypeError) as e:
             warnings.warn(f"Invalid rect arguments: {e}", stacklevel=2)
             # Don't raise, just ignore as per spec for path errors

    def roundRect(self, x: float, y: float, w: float, h: float, radii: float | dict | list | tuple = 0) -> None:
        """Adds a rounded rectangle shape definition to the path."""
        try:
             # Path2D.roundRect handles validation
             self._current_path.roundRect(x, y, w, h, radii)
        except Exception as e:
             # Catch potential errors from Path2D if it raises them
             warnings.warn(f"Error in roundRect path command: {e}", stacklevel=2)


    # --- Path Drawing Operations ---

    def fill(self, path_or_rule: Path2D | str | None = None, fillRule: str | None = None) -> None:
        """Fills the current path or a given path with the current fillStyle, respecting fillRule."""
        target_path = self._current_path # Default to current path
        rule = 'nonzero' # Default fill rule

        # --- Argument Parsing ---
        # Determine which argument is the Path2D (if any) and which is the rule (if any)
        if isinstance(path_or_rule, Path2D):
            target_path = path_or_rule # Use provided path
            # If second arg exists and is string, it's the rule
            if isinstance(fillRule, str):
                rule = fillRule.lower()
            elif fillRule is not None:
                 warnings.warn(f"Invalid fillRule type '{type(fillRule)}' when Path2D provided. Ignoring rule.", stacklevel=2)
            # else: rule remains 'nonzero'
        elif isinstance(path_or_rule, str):
            # First arg is the rule, use current path
            rule = path_or_rule.lower()
            # fillRule (second arg) should be None here, warn if provided erroneously
            if fillRule is not None:
                 warnings.warn("Unexpected second argument when fillRule provided as first argument. Ignoring second arg.", stacklevel=2)
        elif path_or_rule is None:
             # No args, or only fillRule provided. Use current path.
             if isinstance(fillRule, str):
                  rule = fillRule.lower()
             elif fillRule is not None: # e.g., fill(None, 123) -> invalid rule type
                  warnings.warn(f"Invalid fillRule type '{type(fillRule)}'. Using default 'nonzero'.", stacklevel=2)
        else: # Invalid first argument type (e.g., fill(123))
            warnings.warn(f"Invalid first argument type '{type(path_or_rule)}' for fill(). Ignoring call.", stacklevel=2)
            return

        # Validate fill rule string
        if rule not in ('nonzero', 'evenodd'):
            warnings.warn(f"Invalid fillRule '{rule}'. Using default 'nonzero'.", stacklevel=2)
            rule = 'nonzero'

        # Check if path is valid and has content
        if not target_path or not target_path.subpaths:
            # Logger.debug("fill() called with empty path.")
            return # Nothing to fill

        style = self._fillStyle # Get current fill style

        # --- Prepare for Drawing ---
        # Generate geometry suitable for filling using Mesh (needed for gradients and robust stencil)
        path_geometry = self._get_geometry_for_fill(target_path)
        # Get solid color (applies globalAlpha) if applicable
        solid_color_rgba = self._get_current_color('fill')

        # Check if there's anything to draw (visible color or gradient) and geometry exists
        is_gradient = hasattr(style, '_is_kivy_gradient') and style._is_kivy_gradient
        is_visible_solid = solid_color_rgba and solid_color_rgba[3] > 0
        can_draw = (is_gradient or is_visible_solid) and path_geometry

        if not can_draw:
            # Logger.debug("fill() skipped: No visible style or no geometry.")
            return # Nothing to draw

        # --- Perform Fill ---
        self._push_drawing_state() # Apply transform & main clipping context
        with self.canvas:
            # Use Stencil buffer to handle fill rules accurately for both solid and gradient.
            # 1. Draw path geometry into stencil buffer to create fill mask based on rule.
            StencilPush() # Save current stencil state (e.g., from outer clip)
            # Kivy manages glStencilOp internally for mask writing based on rule (implicitly)
            Color(0,0,0,0) # Don't draw color for stencil mask generation
            # Render the Mesh geometry into the stencil buffer
            if path_geometry: # Check geometry exists
                # Mesh requires x,y,u,v vertex format usually, but stencil just needs position.
                # Kivy might handle missing UVs, or we need to add dummy ones.
                # Let's assume Kivy's basic Mesh for stencil works with just pos if no texture bound.
                # We use _get_geometry_for_fill which only provides x,y flattened.
                # Reformat vertices for Kivy Mesh (x,y,u,v) - add dummy UVs
                stencil_mesh_vertices = []
                verts = path_geometry['vertices']
                for i in range(0, len(verts), 2):
                    stencil_mesh_vertices.extend([verts[i], verts[i+1], 0.0, 0.0]) # x,y, u=0,v=0

                self.canvas.add(Mesh(vertices=stencil_mesh_vertices, indices=path_geometry['indices'], mode='triangles'))

            # 2. Set up stencil test (implicitly handled by StencilUse)
            # 3. Enable stencil test and draw the actual fill (gradient or solid color).
            StencilUse() # Use Kivy's default stencil test setup
            if rule == 'evenodd': # Add warning for evenodd rule
                warnings.warn("Fill rule 'evenodd' might not render correctly with default Kivy StencilUse. NonZero rule is better supported.", stacklevel=3)

            if is_gradient:
                # Use the same path_geometry dict (contains x,y vertices) for gradient shader
                self._render_gradient_with_shader(style, path_geometry)
            elif is_visible_solid:
                Color(*solid_color_rgba)
                # Draw rectangle covering the widget clipped by stencil
                # Need widget's bounding box in local coordinates (0,0 to self.size)
                # The transform matrix handles placement on screen.
                Rectangle(pos=(0, 0), size=self.size) # This might need adjustment based on transform origin

            StencilUnUse() # Disable stencil test after drawing fill
            StencilPop() # Restore previous stencil state

        self._pop_drawing_state() # Restore transform & main clipping context


    def stroke(self, path: Path2D | None = None) -> None:
        """Strokes the outline of the current path or a given path."""
        target_path = path if isinstance(path, Path2D) else self._current_path
        # Ignore if path is empty, line width is zero, or stroke style is transparent/invalid
        stroke_color_rgba = self._get_current_color('stroke') # Handles gradient fallback & alpha
        if not target_path or not target_path.subpaths or self._lineWidth <= 0 or \
           not stroke_color_rgba or stroke_color_rgba[3] == 0:
            return

        # --- Draw Stroke ---
        self._push_drawing_state() # Apply transform & clipping
        with self.canvas:
             Color(*stroke_color_rgba) # Set the stroke color
             # Use helper to add Kivy Line instructions for the path segments/shapes
             # _render_path_geometry handles polygons and roundRects in stroke mode.
             # TODO: Curves (bezier, arc) need proper geometry generation for stroking in Path2D.
             #       Current implementation uses placeholders in Path2D, leading to incorrect strokes.
             self._render_path_geometry(target_path, 'stroke')

        self._pop_drawing_state() # Restore transform/clipping


    # --- Clipping Path Operations ---
    def clip(self, path_or_rule: Path2D | str | None = None, fillRule: str | None = None) -> None:
        """Sets the current clipping region based on a path and fill rule."""
        target_path = self._current_path # Default to current path if none provided
        rule = 'nonzero' # Default fill rule

        # --- Argument Parsing (similar to fill()) ---
        if isinstance(path_or_rule, Path2D):
            target_path = path_or_rule # Use provided path
            if isinstance(fillRule, str): rule = fillRule.lower()
            elif fillRule is not None: warnings.warn(f"Invalid clip fillRule type '{type(fillRule)}'. Ignoring rule.", stacklevel=2)
        elif isinstance(path_or_rule, str):
            rule = path_or_rule.lower() # Use rule, use current path
            if fillRule is not None: warnings.warn("Unexpected second argument for clip() when rule provided first.", stacklevel=2)
        elif path_or_rule is None:
             if isinstance(fillRule, str): rule = fillRule.lower() # Use rule, use current path
             elif fillRule is not None: warnings.warn(f"Invalid clip fillRule type '{type(fillRule)}'. Using default 'nonzero'.", stacklevel=2)
        else: # Invalid first arg type
            warnings.warn(f"Invalid first argument type '{type(path_or_rule)}' for clip(). Ignoring call.", stacklevel=2)
            return

        # Validate fill rule
        if rule not in ('nonzero', 'evenodd'):
            warnings.warn(f"Invalid clip fillRule '{rule}'. Using 'nonzero'.", stacklevel=2)
            rule = 'nonzero'

        # --- Set Clipping State ---
        # Create a *deep copy* of the target path to use for clipping.
        # This prevents subsequent modifications to the original path (e.g., self._current_path)
        # from affecting the active clip region.
        if target_path and target_path.subpaths:
            # Path is valid and has content, store a deep copy and the rule
            self._clip_path = copy.deepcopy(target_path)
            self._clip_fill_rule = rule
            # Logger.debug(f"Set clip path with rule '{rule}'.")
        else:
            # If path is empty or invalid, spec implies clip region becomes empty.
            # Setting _clip_path to None effectively disables clipping.
            # Logger.debug("Setting clip path to None (empty path provided).")
            self._clip_path = None
            self._clip_fill_rule = 'nonzero' # Reset rule as well

        # Note: The actual application of the clipping happens in _push_drawing_state/_begin_clip.
        # This method just updates the internal state (_clip_path, _clip_fill_rule).


    # --- Focus Management (Placeholders - No Visual Implementation) ---
    # TODO: Implement focus ring drawing if needed, integrating with Kivy focus.
    def drawFocusIfNeeded(self, element):
         """Draws a focus ring if the element is focused. (Not implemented visually)."""
         # Requires integration with Kivy's focus behavior and potentially platform APIs.
         # element argument in web API refers to HTML element. Need Kivy equivalent?
         warnings.warn("drawFocusIfNeeded() is not implemented visually.", stacklevel=2)

    def scrollPathIntoView(self, path: Path2D | None = None):
         """Scrolls the canvas container so the path is visible. (Not implemented)."""
         # Requires the canvas to be inside a Kivy ScrollView or similar container.
         # Would need to calculate bounding box of the path and interact with ScrollView.
         warnings.warn("scrollPathIntoView() is not implemented.", stacklevel=2)


    # --- Hit Regions (Placeholders - No Implementation) ---
    # TODO: Implement hit region tracking and testing against Kivy touch events.
    def addHitRegion(self, options: dict | None = None):
         """Adds a hit detection region. (Not implemented)."""
         # Requires mapping regions to IDs and handling Kivy touch events (on_touch_down etc.)
         # to check if touch point falls within a registered region's path.
         # Options dict includes: path, fillRule, id, cursor, control, label, role.
         warnings.warn("addHitRegion() is not implemented.", stacklevel=2)

    def removeHitRegion(self, id: str):
         """Removes the hit region with the specified ID. (Not implemented)."""
         warnings.warn("removeHitRegion() is not implemented.", stacklevel=2)

    def clearHitRegions(self):
         """Removes all hit regions. (Not implemented)."""
         warnings.warn("clearHitRegions() is not implemented.", stacklevel=2)


    # --- Point-in-Path / Point-in-Stroke Testing ---
    def isPointInPath(self, path_or_x: Path2D | float, x_or_y: float, y_or_rule: float | str | None = None, fillRule: str = 'nonzero') -> bool:
        """Checks if the given point (in Canvas coordinates) is inside the specified path, respecting transforms and fill rule."""
        target_path = self._current_path
        x, y = 0.0, 0.0
        rule = 'nonzero' # Default rule if not specified otherwise

        # --- Argument Parsing (Robust handling of multiple signatures) ---
        try:
            if isinstance(path_or_x, Path2D):
                # Signature: isPointInPath(path, x, y, fillRule?)
                target_path = path_or_x
                # Check next two args are numbers (x, y)
                x = float(x_or_y) # Can raise ValueError/TypeError
                # y_or_rule holds the y coordinate
                if y_or_rule is None or isinstance(y_or_rule, str):
                    # This means signature was likely (path, x, rule?) which is invalid
                    raise TypeError("Missing Y coordinate for signature isPointInPath(path, x, y, ...)")
                y = float(y_or_rule) # Can raise
                # Fourth arg 'fillRule' (in the method signature) is the potential rule override
                if fillRule is not None and isinstance(fillRule, str): # Check the 4th signature arg explicitly
                    rule = fillRule.lower()
                # Ignore fillRule if not a string

            elif isinstance(path_or_x, (int, float)):
                # Signature: isPointInPath(x, y, fillRule?)
                target_path = self._current_path # Use current path
                x = float(path_or_x) # First arg is x
                y = float(x_or_y)    # Second arg is y
                # Third arg 'y_or_rule' holds the potential rule override string
                if y_or_rule is not None:
                    if isinstance(y_or_rule, str):
                        rule = y_or_rule.lower()
                    else: # Invalid type for rule
                         raise TypeError(f"Invalid fillRule type: {type(y_or_rule)} for signature isPointInPath(x, y, rule)")
                # fillRule (4th arg in method signature) is ignored in this signature path

            else: # First argument invalid type
                raise TypeError("First argument must be Path2D or number (x-coordinate)")

            # Validate coordinates and rule after parsing
            if not (math.isfinite(x) and math.isfinite(y)):
                # Spec: Non-finite points are never in the path.
                return False
            if rule not in ('nonzero', 'evenodd'):
                warnings.warn(f"Invalid fillRule '{rule}' for isPointInPath. Using 'nonzero'.", stacklevel=3)
                rule = 'nonzero' # Default to nonzero if rule is invalid string
            if not target_path or not target_path.subpaths:
                # Spec: Point is never in an empty path.
                return False

        except (ValueError, TypeError) as e:
            # Catch float conversion errors or explicit TypeErrors from parsing
            warnings.warn(f"Invalid arguments for isPointInPath: {e}. Returning False.", stacklevel=2)
            return False
        # --- End Argument Parsing ---

        # --- Point-in-Polygon Test using Stencil Buffer ---
        # This is generally robust for complex paths and consistent with fill() rendering.
        # Requires rendering the path to an offscreen FBO with a stencil buffer.
        # Performance: Creating FBO per call can be slow. Consider pooling/reuse if called often.
        hit = False
        temp_fbo = None
        canvas_w_int, canvas_h_int = 0, 0 # Canvas size as integers
        try:
             # Get current canvas integer dimensions for FBO size
             canvas_w_int, canvas_h_int = int(self.width), int(self.height)
             if canvas_w_int <= 0 or canvas_h_int <= 0:
                 return False # Cannot perform test if canvas area is zero.

             # --- Use Temporary FBO with Stencil for Hit Test ---
             # Create a new FBO per call for isolation. Size matches current canvas.
             temp_fbo = Fbo(size=(canvas_w_int, canvas_h_int), with_stencilbuffer=True)
             if not temp_fbo: raise RuntimeError("Failed to create temporary FBO for isPointInPath")

             with temp_fbo:
                 ClearColor(0, 0, 0, 0)
                 ClearBuffers(clear_color=True, clear_stencil=True) # Clear FBO stencil
                 # Add necessary context instructions to the FBO canvas

                 temp_fbo.add(PushMatrix())
                 baseInstr = MatrixInstruction()
                 cloneMatrix = Matrix()
                 cloneMatrix.set(flat = self._base_matrix.get())
                 baseInstr.matrix = cloneMatrix
                 temp_fbo.add(baseInstr)
                 userInstr = MatrixInstruction()
                 cloneMatrix = Matrix()
                 cloneMatrix.set(flat = self._user_matrix.get())
                 userInstr.matrix = cloneMatrix
                 temp_fbo.add(userInstr)
                 temp_fbo.add(self._update_normal_matrix_instr)

                 # Draw path geometry into stencil buffer based on fill rule
                 temp_fbo.add(StencilPush())
                 temp_fbo.add(Color(0,0,0,0)) # Draw transparent for mask
                 # Generate geometry (only need x,y flattened)
                 hit_test_geometry = self._get_geometry_for_fill(target_path)
                 if hit_test_geometry:
                      # Format for Mesh (x,y,u,v)
                      mesh_verts = []
                      verts = hit_test_geometry['vertices']
                      for i in range(0, len(verts), 2):
                          mesh_verts.extend([verts[i], verts[i+1], 0.0, 0.0])
                      temp_fbo.add(Mesh(vertices=mesh_verts, indices=hit_test_geometry['indices'], mode='triangles'))
                 else:
                      temp_fbo.add(StencilPop()); temp_fbo.add(PopMatrix()) # Clean up FBO canvas
                      warnings.warn("isPointInPath: Failed to generate geometry for hit test.", stacklevel=3)
                      return False # Treat as no hit if geometry fails

                 # Balance the StencilPush even after drawing
                 temp_fbo.add(StencilPop())
                 # Balance the PushMatrix
                 temp_fbo.add(PopMatrix())

             # --- Check Stencil Value at Point (Indirect Method) ---
             temp_fbo.bind()
             try:
                 # Draw a 1x1 marker pixel gated by stencil test
                 # (Need to re-add state instructions inside FBO's canvas context)
                 marker_color_instr = Color(1, 0, 1, 1) # Magenta
                 temp_fbo.add(PushMatrix())
                 baseInstr = MatrixInstruction()
                 cloneMatrix = Matrix()
                 cloneMatrix.set(flat = self._base_matrix.get())
                 baseInstr.matrix = cloneMatrix
                 temp_fbo.add(baseInstr)
                 userInstr = MatrixInstruction()
                 cloneMatrix = Matrix()
                 cloneMatrix.set(flat = self._user_matrix.get())
                 userInstr.matrix = cloneMatrix
                 temp_fbo.add(userInstr)
                 temp_fbo.add(self._update_normal_matrix_instr)
                 pixel_rect_instr = Rectangle(pos=(x, y), size=(1, 1)) # Draw at GL coordinates

                 temp_fbo.canvas.add(StencilPush()) # Push state for test
                 temp_fbo.canvas.add(StencilUse()) # Enable test based on mask drawn earlier
                 if rule == 'evenodd': # Add warning
                      warnings.warn("isPointInPath rule 'evenodd' might not test correctly with default Kivy StencilUse.", stacklevel=3)
                 temp_fbo.canvas.add(marker_color_instr)
                 temp_fbo.canvas.add(pixel_rect_instr)
                 temp_fbo.canvas.add(StencilUnUse()) # Disable test
                 temp_fbo.canvas.add(StencilPop()) # Pop test state

                 temp_fbo.draw() # Execute drawing instructions on FBO

                 # Read the pixel color
                 pixel_data = glReadPixels(x, y, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE)

                 # Check if pixel data matches marker color
                 if pixel_data and len(pixel_data) == 4:
                      # Check for Magenta (approx values due to potential blending/precision)
                      if pixel_data[0] > 200 and pixel_data[1] < 50 and pixel_data[2] > 200 and pixel_data[3] > 200:
                          hit = True

             except Exception as read_error:
                  warnings.warn(f"isPointInPath: Stencil read simulation failed: {read_error}", stacklevel=2)
                  # Ensure FBO state is potentially cleaned up in finally block
             finally:
                  # Clean up instructions added for testing *within* the try block
                  # (This cleanup logic might need refinement based on Kivy's FBO canvas handling)
                  # It might be better to clear and redraw the FBO canvas if reusing.
                  # Since we create a new FBO each time, maybe just releasing is enough.
                  temp_fbo.add(PopMatrix())
                  if temp_fbo.is_bound(): temp_fbo.release()

        except Exception as e:
            warnings.warn(f"Error during isPointInPath execution: {e}", stacklevel=2)
            hit = False # Default to false on any error during the process
        finally:
             # Clean up temporary FBO resources if created
             if temp_fbo:
                 # Kivy FBOs are supposed to clean up textures/buffers on GC
                 pass

        return hit


    def isPointInStroke(self, path_or_x: Path2D | float, x_or_y: float, y: float | None = None) -> bool:
        """Checks if the given point is within the stroke area of the specified path."""
        # This is significantly more complex than isPointInPath.
        # TODO: Implement robust stroke hit testing if needed.
        warnings.warn("isPointInStroke() is not implemented. Returning False.", stacklevel=2)

        # --- Basic Argument Parsing (similar to isPointInPath, no fillRule) ---
        target_path = self._current_path
        x, y_coord = 0.0, 0.0 # Renamed 'y' to avoid conflict with method signature 'y'
        try:
             if isinstance(path_or_x, Path2D): # isPointInStroke(path, x, y)
                  target_path = path_or_x
                  x = float(x_or_y)
                  if y is None: raise TypeError("Missing Y coordinate for signature isPointInStroke(path, x, y)")
                  y_coord = float(y) # Use the signature 'y' argument
             elif isinstance(path_or_x, (int, float)): # isPointInStroke(x, y)
                  target_path = self._current_path
                  x = float(path_or_x)
                  y_coord = float(x_or_y) # Use the second argument as y
                  if y is not None: warnings.warn("Unexpected third argument for isPointInStroke(x, y).", stacklevel=2)
             else: raise TypeError("First argument must be Path2D or number (x-coordinate)")

             # Validate coordinates and state
             if not (math.isfinite(x) and math.isfinite(y_coord)): return False
             if not target_path or not target_path.subpaths or self._lineWidth <= 0: return False

        except (ValueError, TypeError) as e:
             warnings.warn(f"Invalid arguments for isPointInStroke: {e}. Returning False.", stacklevel=2)
             return False
        # --- End Argument Parsing ---

        # Actual hit testing logic for stroke area is missing.
        return False


    # --- Image Drawing ---
    def loadTexture(self, source: str | Texture | TextureRegion | CoreImage | ImageData | object) -> Texture | TextureRegion | None:
        """Helper to load a Kivy texture from various sources, with basic caching for strings."""
        # Returns a Kivy Texture or TextureRegion, or None on failure.
        tex = None
        cache_key = None # Used only for string sources

        try:
            # 1. Handle string source (filepath/URL) with caching
            if isinstance(source, str):
                cache_key = source
                if source in self._texture_cache:
                    cached_tex = self._texture_cache[source]
                    # Check if cached texture is still valid (allocated in GPU)
                    if cached_tex:
                        # Logger.debug(f"Canvas: Using cached texture for: {source}")
                        return cached_tex
                    else:
                        # Remove invalid entry from cache
                        Logger.debug(f"Canvas: Removing invalid texture from cache: {source}")
                        del self._texture_cache[source]

                # If not cached or invalid, load using Kivy CoreImage
                # keep_data=False allows Kivy to potentially free CPU buffer after GPU upload.
                # Logger.debug(f"Canvas: Loading texture from source: {source}")
                img = CoreImage(source, keep_data=False)
                tex = img.texture # Get the Kivy Texture object
                if tex is None:
                     # CoreImage might return None if loading failed internally
                     raise ValueError(f"CoreImage failed to load texture from '{source}' (file not found or invalid format?).")

            # 2. Handle existing Kivy Texture/TextureRegion
            elif isinstance(source, (Texture, TextureRegion)):
                tex = source # Use directly
                # Don't cache externally provided textures unless given a key? Keep it simple.

            # 3. Handle Kivy CoreImage object
            elif isinstance(source, CoreImage):
                tex = source.texture
                if tex is None:
                     raise ValueError("Provided CoreImage object has no valid texture.")

            # 4. Handle our ImageData class
            elif isinstance(source, ImageData):
                # Get texture via the ImageData property (handles internal caching & V-flip)
                tex = source.texture # Access the @property getter
                # No warning needed here, ImageData.texture handles errors/warnings

            # 5. Handle generic objects with a 'texture' attribute (Duck typing)
            elif hasattr(source, 'texture'):
                tex_attr = getattr(source, 'texture')
                if isinstance(tex_attr, (Texture, TextureRegion)):
                    tex = tex_attr
                else:
                    raise TypeError(f"Source object's .texture attribute is not a Kivy Texture/TextureRegion: {type(tex_attr)}")
            else:
                # Unsupported source type
                raise TypeError(f"Unsupported image source type for drawImage: {type(source)}")

            # --- Final Validation & Caching ---
            # Ensure we got a valid, allocated texture object
            if tex:
                 if cache_key: # Cache only if loaded from string source
                     # Logger.debug(f"Canvas: Caching texture for: {cache_key}")
                     self._texture_cache[cache_key] = tex
                 return tex
            else:
                 # Loading failed or texture is invalid/unallocated
                 # Warning should have been issued by CoreImage or ImageData.texture
                 warnings.warn(f"Failed to get valid/allocated texture from source: {source}", stacklevel=3)
                 return None

        except Exception as e:
            # Log and warn about texture loading failures
            warnings.warn(f"Failed to load texture from source '{source}': {e}", stacklevel=2)
            # Ensure cache is clean if loading failed for a cached key
            if cache_key and cache_key in self._texture_cache:
                 try: del self._texture_cache[cache_key]
                 except KeyError: pass
            return None

    def drawImage(self, image_source, *args) -> None:
        """Draws the given image source onto the canvas, handling various signatures."""
        # --- Load Texture ---
        texture = self.loadTexture(image_source)
        if not texture:
            # Warning already issued by loadTexture if loading failed
            # Logger.warning(f"drawImage failed: Could not load texture from source.")
            return

        # Get texture dimensions (use texture.size, not image_source directly)
        tex_w, tex_h = texture.size
        if tex_w <= 0 or tex_h <= 0:
            warnings.warn(f"drawImage failed: Texture has zero dimensions ({tex_w}x{tex_h}).", stacklevel=3)
            return

        # --- Argument Parsing ---
        # Determine signature based on number of numeric arguments:
        # drawImage(image, dx, dy) -> 2 args
        # drawImage(image, dx, dy, dw, dh) -> 4 args
        # drawImage(image, sx, sy, sw, sh, dx, dy, dw, dh) -> 8 args
        num_args = len(args)
        # Default values: draw full source rect (0,0,tex_w,tex_h) at dest (dx,dy) with original size (tex_w, tex_h)
        sx, sy, sw, sh = 0.0, 0.0, float(tex_w), float(tex_h) # Source rect (TL origin)
        dx, dy = 0.0, 0.0 # Destination position (TL origin)
        dw, dh = float(tex_w), float(tex_h) # Destination size

        try:
            if num_args == 2: # drawImage(image, dx, dy)
                dx, dy = map(float, args)
                # dw, dh remain tex_w, tex_h
            elif num_args == 4: # drawImage(image, dx, dy, dw, dh)
                dx, dy, dw, dh = map(float, args)
            elif num_args == 8: # drawImage(image, sx, sy, sw, sh, dx, dy, dw, dh)
                sx, sy, sw, sh, dx, dy, dw, dh = map(float, args)
            elif num_args == 0:
                 # Allow drawImage(img) to draw at (0,0) with original size? Spec unclear. Let's require coords.
                 raise TypeError(f"drawImage requires at least 2 numeric arguments (dx, dy) after image source.")
            else: # Invalid number of arguments (1, 3, 5, 6, 7)
                 raise TypeError(f"drawImage requires 2, 4, or 8 numeric arguments after image source, got {num_args}")

            # --- Validate Arguments ---
            all_coords = [sx, sy, sw, sh, dx, dy, dw, dh]
            if not all(math.isfinite(c) for c in all_coords):
                 warnings.warn(f"drawImage failed: Non-finite coordinate values detected.", stacklevel=3)
                 return
            # Spec: If dw or dh is 0, draw nothing. Also if sw or sh is 0.
            # Handle negative sw/sh by adjusting sx/sy? No, spec says use absolute for source rect?
            # Let's clamp sw/sh >= 0. If sw/sh becomes 0, skip draw.
            # Negative dw/dh means flip? Kivy Rectangle handles negative size by adjusting pos.
            if sw <= 0 or sh <= 0 or dw == 0 or dh == 0:
                 # Logger.debug("drawImage skipped due to zero width/height in source or destination.")
                 return

            # Adjust destination x/y if destination w/h are negative (Kivy Rectangle behavior)
            draw_dx = dx if dw >= 0 else dx + dw
            draw_dy = dy if dh >= 0 else dy + dh
            draw_dw = abs(dw)
            draw_dh = abs(dh)

            # --- Calculate Kivy Texture Coordinates (UV) ---
            # Map the Canvas source rectangle (sx, sy, sw, sh) with TL origin
            # to Kivy texture coordinates (u, v) with BL origin [0, 1].
            # Assumes texture (even from ImageData) is oriented correctly for Kivy draw (BL origin).
            # Kivy V = 1.0 - (Canvas Y / Texture Height)

            # Calculate UV coords for the corners of the source rectangle
            u0 = sx / tex_w            # Left edge U
            u1 = (sx + sw) / tex_w      # Right edge U
            # Calculate V for the top (sy) and bottom (sy + sh) edges in Canvas space
            v_top_canvas = sy / tex_h
            v_bottom_canvas = (sy + sh) / tex_h
            # Convert to Kivy V coords (Bottom-Left origin)
            v0_kivy = 1.0 - v_bottom_canvas # Kivy V for bottom edge
            v1_kivy = 1.0 - v_top_canvas    # Kivy V for top edge

            # Kivy Rectangle `tex_coords` expect order: BL, BR, TR, TL as (u,v) pairs flattened
            tex_coords = [
                u0, v0_kivy,  # Bottom-Left vertex -> (sx, sy+sh) maps to (u0, v0_kivy)
                u1, v0_kivy,  # Bottom-Right vertex -> (sx+sw, sy+sh) maps to (u1, v0_kivy)
                u1, v1_kivy,  # Top-Right vertex -> (sx+sw, sy) maps to (u1, v1_kivy)
                u0, v1_kivy   # Top-Left vertex -> (sx, sy) maps to (u0, v1_kivy)
            ]

        except (ValueError, TypeError) as e:
            warnings.warn(f"Invalid numeric arguments for drawImage: {args} -> {e}. Ignoring call.", stacklevel=2)
            return

        # --- Draw using Kivy Rectangle ---
        self._push_drawing_state() # Apply transforms and clipping
        with self.canvas:
            # Set texture filtering based on smoothing settings
            filter_mode = self._get_smoothing_filter()
            texture.mag_filter = filter_mode
            texture.min_filter = filter_mode

            # Apply globalAlpha via Kivy's Color instruction.
            # Assume base texture color is white (no tinting by default).
            Color(1, 1, 1, self._globalAlpha)

            # Draw the Kivy Rectangle instruction.
            # Position (draw_dx, draw_dy) and size (draw_dw, draw_dh) are in Canvas coordinates.
            # The MatrixInstruction applied by _push_drawing_state handles the
            # transformation to Kivy's coordinate system for rendering.
            Rectangle(pos=(draw_dx, draw_dy), size=(draw_dw, draw_dh),
                      texture=texture,
                      tex_coords=tex_coords) # Pass the calculated UVs

        self._pop_drawing_state() # Restore state


    # --- Pixel Manipulation ---
    def createImageData(self, sw_or_data: int | float | ImageData, sh: int | float | None = None) -> ImageData:
        """Creates a new ImageData object, either blank or copying existing data."""
        width, height, data = 0, 0, b'' # Initialize defaults
        colorSpace = "srgb" # Initialize default colorSpace
        try:
            if isinstance(sw_or_data, ImageData):
                # Signature: createImageData(existing_imageData) -> Copy
                # Spec requires TypeError if source is not ImageData
                s = sw_or_data
                width = s.width
                height = s.height
                # Create a copy of the data buffer
                data = bytes(s.data) if s.data else b''
                # Copy color space as well
                colorSpace = s.colorSpace
            elif isinstance(sw_or_data, (int, float)) and isinstance(sh, (int, float)):
                # Signature: createImageData(sw, sh) -> Blank RGBA
                # Spec: Arguments are doubles, throw if non-finite. Use abs value.
                sw_f, sh_f = float(sw_or_data), float(sh)
                if not (math.isfinite(sw_f) and math.isfinite(sh_f)):
                     # Spec: throw RangeError, simulate with ValueError or TypeError
                     raise ValueError("Width and height must be finite numbers.")

                # Convert potentially float width/height to absolute integer
                width = abs(int(sw_f))
                height = abs(int(sh_f))

                if width <= 0 or height <= 0:
                     width, height = 0, 0 # Result is 0x0 ImageData
                     data = b''
                else:
                     # Create blank (transparent black) data buffer: R=0, G=0, B=0, A=0
                     num_pixels = width * height
                     data = b'\x00\x00\x00\x00' * num_pixels
                colorSpace = "srgb" # Default for new blank data
            else:
                 # Invalid arguments combination
                 raise TypeError("Invalid arguments: require (ImageData) or (width, height)")

            # Return the new ImageData object
            return ImageData(width, height, data, colorSpace=colorSpace)

        except (ValueError, TypeError) as e:
            # Re-raise TypeError for spec compliance on invalid args/types?
            # Or just propagate the specific error. Let's propagate.
             raise e


    def getImageData(self, sx: int | float, sy: int | float, sw: int | float, sh: int | float, *, settings: dict | None = None) -> ImageData:
        """Extracts pixel data from the canvas region into an ImageData object."""
        # Handle settings argument (new spec vs old colorSpace)
        colorSpace = "srgb" # Default
        premultipliedAlpha = False # Default? Spec is complex here. Assume false for output.
        if settings:
             colorSpace = settings.get("colorSpace", "srgb")
             premultipliedAlpha = settings.get("premultipliedAlpha", False) # Doesn't affect read, maybe affects output interpretation? Ignore for now.

        # Validate colorSpace - Kivy doesn't support others yet
        if colorSpace != "srgb":
            warnings.warn(f"getImageData colorSpace '{colorSpace}' is ignored; using srgb.", stacklevel=2)
            colorSpace = "srgb" # Force srgb

        # Get current canvas integer dimensions (needed for clamping and FBO)
        canvas_w_int, canvas_h_int = int(self.width), int(self.height)
        if canvas_w_int <= 0 or canvas_h_int <= 0:
             # If canvas has no area, cannot get data. Return empty ImageData.
             # Spec says return 0x0 if requested sw/sh is 0, what if canvas is 0? Let's return 0x0.
             # Try to get requested size, default to 0 if invalid.
             try: out_w, out_h = abs(int(sw)), abs(int(sh))
             except: out_w, out_h = 0, 0
             if out_w <= 0 or out_h <= 0: out_w, out_h = 0, 0
             return ImageData(out_w, out_h, None, colorSpace=colorSpace)

        # Validate input arguments (allow float, convert to int, check finite)
        try:
            sx_f, sy_f, sw_f, sh_f = map(float, [sx, sy, sw, sh])
            if not all(math.isfinite(v) for v in [sx_f, sy_f, sw_f, sh_f]):
                 raise ValueError("Arguments must be finite numbers.") # Spec: RangeError
            # Convert to integers for pixel coordinates
            sx_i, sy_i = int(sx_f), int(sy_f)
            sw_i, sh_i = int(sw_f), int(sh_f)
        except (ValueError, TypeError) as e:
            # Spec: RangeError if non-finite, TypeError if wrong type. Simulate with ValueError/TypeError.
            raise ValueError(f"Invalid arguments for getImageData: {e}") from e

        # --- Calculate Read Area and Output Size ---
        # Output ImageData dimensions are based on absolute requested size
        out_w = abs(sw_i)
        out_h = abs(sh_i)
        # Spec: If sw or sh is 0, return ImageData with 0 width/height.
        if out_w == 0 or out_h == 0:
            return ImageData(0, 0, None, colorSpace=colorSpace) # Create empty 0x0

        # Determine the actual rectangle to read from the canvas (in Canvas TL coordinates)
        # Handle negative sw/sh by adjusting the top-left corner (sx, sy)
        read_x_canvas = sx_i if sw_i >= 0 else sx_i + sw_i # sx + sw = left edge if sw negative
        read_y_canvas = sy_i if sh_i >= 0 else sy_i + sh_i # sy + sh = top edge if sh negative
        read_w_canvas = out_w
        read_h_canvas = out_h

        # Intersect the requested read rectangle with the actual canvas bounds [0, 0, canvas_w, canvas_h]
        intersect_x0 = max(read_x_canvas, 0)
        intersect_y0 = max(read_y_canvas, 0)
        intersect_x1 = min(read_x_canvas + read_w_canvas, canvas_w_int)
        intersect_y1 = min(read_y_canvas + read_h_canvas, canvas_h_int)

        # Calculate the dimensions of the actual area read from the canvas (clamped)
        clamped_read_w = max(0, intersect_x1 - intersect_x0)
        clamped_read_h = max(0, intersect_y1 - intersect_y0)

        # If the intersection is empty, return blank ImageData of the originally requested size
        if clamped_read_w <= 0 or clamped_read_h <= 0:
            return ImageData(out_w, out_h, None, colorSpace=colorSpace) # Create blank

        # --- Read Pixels using FBO ---
        pixels = None # This will hold the raw bytes read from OpenGL (RGBA order)
        fbo_read_success = False
        temp_fbo = None # Use a temporary FBO to avoid interfering with self._fbo state
        try:
            fbo_size = (canvas_w_int, canvas_h_int)
            # Logger.debug(f"getImageData: Creating temp FBO {fbo_size}x{fbo_size}")
            temp_fbo = Fbo(size=fbo_size, with_stencilbuffer=True) # Need stencil for clip/fill rules
            if not temp_fbo: raise RuntimeError("Failed to create temporary FBO for getImageData")

            # Bind the FBO and render the current main canvas content into it
            with temp_fbo:
                ClearColor(0, 0, 0, 0) # Clear to transparent black
                ClearBuffers(clear_color=True, clear_stencil=True)
                # Replicate main canvas instructions into the FBO's canvas
                # This assumes instructions are reusable. If they modify state internally, might cause issues.
                # Using canvas.children assumes flat structure. Group might need recursion.
                for instr in self.canvas.children:
                    # Avoid adding the FBO itself or instructions already part of it?
                    # This replication might be fragile. A better way might be needed.
                    # Let's assume shallow copy/add works for now.
                    temp_fbo.add(instr)

            # Bind the FBO again (outside 'with') to make it the current framebuffer for reading
            temp_fbo.bind()

            # Calculate read position in FBO/OpenGL coordinates (Bottom-Left origin)
            # We read the 'clamped' rectangle calculated earlier.
            gl_read_x = int(intersect_x0) # X is same in Kivy GL coords as Canvas coords
            # Kivy Y = FBO Height - Canvas Y (Top) - Read Height
            # glReadPixels needs the Y coord of the *bottom* edge of the read rectangle in GL coords.
            # Bottom edge Y in Canvas Coords = intersect_y1
            # Bottom edge Y in GL Coords = FBO Height - intersect_y1
            gl_read_y = int(fbo_size[1] - intersect_y1)
            gl_read_w = int(clamped_read_w)
            gl_read_h = int(clamped_read_h)

            # Read pixels from the currently bound FBO (OpenGL command)
            # Reads into a bytes object. Format is RGBA, type unsigned byte.
            pixels = glReadPixels(gl_read_x, gl_read_y, gl_read_w, gl_read_h,
                                  GL_RGBA, GL_UNSIGNED_BYTE)
            fbo_read_success = True # Mark as success if glReadPixels completes without error

        except Exception as e:
            warnings.warn(f"getImageData failed during FBO rendering or reading: {e}", stacklevel=2)
            # Return blank ImageData on error
            return ImageData(out_w, out_h, None, colorSpace=colorSpace)
        finally:
            # Ensure FBO is released and cleaned up
            if temp_fbo:
                 if temp_fbo.is_bound(): temp_fbo.release()
                 # temp_fbo.clear() # Clear instructions added to FBO - might be unsafe if instructions are shared
                 # Delete FBO resources? Rely on GC for now.
                 # temp_fbo.delete_fbo() # Not directly exposed

        # --- Process Read Pixels into Output ImageData Buffer ---
        if not fbo_read_success or pixels is None:
             warnings.warn("getImageData: glReadPixels did not return data or failed.", stacklevel=2)
             return ImageData(out_w, out_h, None, colorSpace=colorSpace)

        # Verify size of returned pixel data
        expected_read_bytes = clamped_read_w * clamped_read_h * 4
        if len(pixels) != expected_read_bytes:
             warnings.warn(f"getImageData: glReadPixels returned unexpected data size ({len(pixels)} bytes, expected {expected_read_bytes}).", stacklevel=2)
             # Return blank data if size is wrong
             return ImageData(out_w, out_h, None, colorSpace=colorSpace)

        # Create the final output buffer, initialized to transparent black
        # Use bytearray for efficient modification.
        final_data_arr = bytearray(out_w * out_h * 4)

        # Calculate where the read data (from the clamped rect) should be placed within the output buffer
        # Offset is relative to the top-left of the requested rect (read_x_canvas, read_y_canvas)
        target_offset_x = int(intersect_x0 - read_x_canvas) # Column offset in output pixels
        target_offset_y = int(intersect_y0 - read_y_canvas) # Row offset in output pixels

        # Check if offsets are valid (should always be >= 0 if logic is correct)
        if target_offset_x < 0 or target_offset_y < 0:
            warnings.warn(f"getImageData: Internal error - negative target offset calculated ({target_offset_x}, {target_offset_y}).", stacklevel=3)
            return ImageData(out_w, out_h, None, colorSpace=colorSpace) # Return blank on internal error

        # Copy pixel data row by row from 'pixels' (clamped read data) into 'final_data_arr'
        read_data_stride = clamped_read_w * 4 # Bytes per row in glReadPixels data (source)
        final_data_stride = out_w * 4         # Bytes per row in output buffer (destination)

        for row_idx in range(clamped_read_h): # Iterate through rows of actual read data
            # Calculate start byte index for the current row in the source 'pixels' buffer
            src_row_start_byte = row_idx * read_data_stride
            # Calculate start byte index for the target position in the 'final_data_arr' buffer
            # Target Row = target_offset_y + row_idx
            # Target Col = target_offset_x
            dst_row_start_byte = (target_offset_y + row_idx) * final_data_stride
            dst_col_start_byte = target_offset_x * 4 # 4 bytes per pixel (RGBA)
            dst_start_byte = dst_row_start_byte + dst_col_start_byte

            # Calculate end byte index for source and destination slices
            src_row_end_byte = src_row_start_byte + read_data_stride
            dst_end_byte = dst_start_byte + read_data_stride # Copy 'read_data_stride' bytes

            # Ensure indices are within bounds (safety check)
            if 0 <= dst_start_byte < len(final_data_arr) and dst_end_byte <= len(final_data_arr) and \
               0 <= src_row_start_byte < len(pixels) and src_row_end_byte <= len(pixels):
                 # Copy the row data bytes from source slice to destination slice
                 final_data_arr[dst_start_byte : dst_end_byte] = \
                     pixels[src_row_start_byte : src_row_end_byte]
            else:
                 # This should not happen if clamping/offset logic is correct
                 warnings.warn(f"getImageData: Calculated pixel copy indices out of bounds (row {row_idx}). Skipping row.", stacklevel=3)

        # Convert the final bytearray to bytes and create the ImageData object
        # The data buffer now contains the requested rectangle, with parts outside the canvas
        # filled with transparent black, and parts inside filled with actual pixel data.
        return ImageData(out_w, out_h, bytes(final_data_arr), colorSpace=colorSpace)


    def putImageData(self, image_data: ImageData, dx: int | float, dy: int | float, dirtyX: int | float = 0, dirtyY: int | float = 0, dirtyWidth: int | float | None = None, dirtyHeight: int | float | None = None) -> None:
        """Draws pixel data from an ImageData object onto the canvas."""
        # Spec: Ignores transforms, shadows, globalAlpha, clipping, composite ops.
        # This Kivy implementation currently deviates from this and IS affected by state.

        # --- Input Validation ---
        if not isinstance(image_data, ImageData):
            raise TypeError("First argument must be an ImageData object.")

        img_w, img_h = image_data.width, image_data.height
        # If ImageData is empty (0 size or no data), there's nothing to draw
        if img_w <= 0 or img_h <= 0 or not image_data.data:
            # Logger.debug("putImageData skipped: Source ImageData is empty.")
            return

        # Get the Kivy texture associated with the ImageData
        # ImageData.texture handles creation and ensures correct vertical orientation for Kivy.
        texture = image_data.texture
        if not texture:
            warnings.warn("putImageData failed: Could not get texture from ImageData.", stacklevel=2)
            return

        # Validate coordinates and dirty rect parameters (use ints after checking finite floats)
        try:
            dx_f, dy_f = float(dx), float(dy)
            dirtyX_f = float(dirtyX); dirtyY_f = float(dirtyY)
            # Default dirtyWidth/Height to the full image dimensions if None
            dirtyWidth_f = float(img_w) if dirtyWidth is None else float(dirtyWidth)
            dirtyHeight_f = float(img_h) if dirtyHeight is None else float(dirtyHeight)

            # Check for non-finite values
            all_vals = [dx_f, dy_f, dirtyX_f, dirtyY_f, dirtyWidth_f, dirtyHeight_f]
            if not all(math.isfinite(v) for v in all_vals):
                 raise ValueError("Arguments must be finite numbers.")

            # Convert to integers for pixel manipulation/coordinates
            dx_i, dy_i = int(dx_f), int(dy_f)
            dirtyX_i, dirtyY_i = int(dirtyX_f), int(dirtyY_f)
            dirtyWidth_i, dirtyHeight_i = int(dirtyWidth_f), int(dirtyHeight_f)

            # --- Clamp Dirty Rect to Source ImageData Bounds ---
            # sx, sy define the top-left corner of the sub-rectangle *within the ImageData* to read from.
            # Clamp dirtyX/Y to be within [0, img_w/h]
            sx = max(0, dirtyX_i)
            sy = max(0, dirtyY_i)
            sx = min(sx, img_w) # Start pos shouldn't exceed image dims
            sy = min(sy, img_h)

            # sw, sh define the dimensions of the sub-rectangle to copy. Clamp >= 0.
            sw = max(0, dirtyWidth_i)
            sh = max(0, dirtyHeight_i)
            # Ensure width/height don't extend beyond bounds starting from sx, sy
            sw = min(sw, img_w - sx)
            sh = min(sh, img_h - sy)

        except (ValueError, TypeError) as e:
            # Propagate as appropriate error type
            raise ValueError(f"Invalid numeric arguments for putImageData: {e}") from e

        # If the dirty rectangle has zero area after clamping, do nothing
        if sw <= 0 or sh <= 0:
            # Logger.debug("putImageData skipped: Dirty rectangle has zero area after clamping.")
            return

        # --- Calculate Texture Coordinates (UV) for the Dirty Rect ---
        # Source rect (sx, sy, sw, sh) is relative to ImageData (TL origin).
        # ImageData.texture was flipped vertically, so UV calculation needs to match Kivy's BL origin.
        # Map TL-based (sx, sy, sw, sh) from ImageData onto the BL-origin Kivy texture.
        u0 = sx / img_w                  # Left U
        u1 = (sx + sw) / img_w            # Right U
        v_bottom = (img_h - (sy + sh)) / img_h # Kivy V for bottom edge (sy+sh)
        v_top = (img_h - sy) / img_h          # Kivy V for top edge (sy)

        # Kivy Rectangle tex_coords: BL, BR, TR, TL (u,v pairs flattened)
        tex_coords = [
            u0, v_bottom, u1, v_bottom, u1, v_top, u0, v_top
        ]

        # --- Draw the Texture Portion ---
        # WARNING: This implementation deviates from the spec regarding ignoring canvas state.
        # It WILL be affected by current transform, clipping, globalAlpha (partially), etc.
        # A fully compliant version is much harder (save/reset/restore GL state or use FBO).
        warnings.warn("putImageData in this implementation IS affected by canvas transform, clipping, and potentially other state (deviation from spec).", stacklevel=2)

        # Draw directly, acknowledging the deviation.
        self._push_drawing_state() # Applies current transform and clip
        with self.canvas:
            # Use nearest neighbor filtering for pixel-perfect drawing (spec behavior)
            texture.mag_filter = 'nearest'
            texture.min_filter = 'nearest'

            # Draw with full opacity, ignoring context's globalAlpha (partial spec compliance)
            Color(1, 1, 1, 1)

            # Draw the specified portion (sw, sh) of the texture at canvas coords (dx_i, dy_i).
            # pos/size are in Canvas coordinates. MatrixInstruction handles transform.
            Rectangle(pos=(dx_i, dy_i), size=(sw, sh), # Use dimensions of the dirty rect
                      texture=texture,
                      tex_coords=tex_coords) # Use UVs calculated for the dirty rect

        self._pop_drawing_state() # Restore transform/clip


    # --- Font Registration Helper ---
    def regFont(self, name: str, path_regular: str, path_bold: str | None = None, path_italic: str | None = None, path_bolditalic: str | None = None) -> None:
        """Registers a font family with Kivy's LabelBase for use in this context."""
        try:
            Logger.info(f"Canvas: Registering font '{name}' -> Regular: '{path_regular}'")
            LabelBase.register(
                name=name,
                fn_regular=path_regular,
                fn_bold=path_bold,
                fn_italic=path_italic,
                fn_bolditalic=path_bolditalic
            )
            # Force update derived font properties IF the registered font matches the current family
            # AND if the context is already past the initial scheduled update.
            # We can check if _font_name differs from the helper's current best guess.
            current_kivy_name = self._font_helper.kivy_font_name # Get potentially updated name
            if name in self._font_helper.font_family and self._font_name != current_kivy_name:
                 Logger.debug(f"Canvas: Font '{name}' registered and matches current family. Forcing property update.")
                 # It should be safe to call this now as regFont is typically called after init
                 self._update_derived_font_properties()
        except Exception as e:
             warnings.warn(f"Failed to register font '{name}' with Kivy: {e}", stacklevel=2)


    # --- Gradient Creation Factory Methods ---
    # These methods perform validation and return gradient objects.

    def createLinearGradient(self, x0: float, y0: float, x1: float, y1: float) -> LinearGradient:
        """Creates a LinearGradient object."""
        try:
             # Spec: Throw TypeError if arguments are non-finite.
             coords = [x0, y0, x1, y1]
             if not all(math.isfinite(float(v)) for v in coords):
                 raise TypeError("Gradient coordinates must be finite numbers.")
             # Return the gradient object instance
             return LinearGradient(x0, y0, x1, y1)
        except (ValueError, TypeError) as e:
             # Reraise TypeError for spec compliance (covers non-finite and non-numeric)
             raise TypeError(f"Invalid arguments for createLinearGradient: {e}") from e

    def createRadialGradient(self, x0: float, y0: float, r0: float, x1: float, y1: float, r1: float) -> RadialGradient:
        """Creates a RadialGradient object."""
        try:
             # Spec: Throw TypeError if coords/radii non-finite. ValueError (like IndexSizeError) if radius negative.
             coords = [x0, y0, r0, x1, y1, r1]
             if not all(math.isfinite(float(v)) for v in coords):
                 raise TypeError("Gradient coordinates and radii must be finite numbers.")
             # Check for negative radii
             if float(r0) < 0 or float(r1) < 0:
                 raise ValueError("Gradient radius cannot be negative.") # Simulates IndexSizeError
             # If validation passes, create the object
             return RadialGradient(x0, y0, r0, x1, y1, r1)
        except (ValueError, TypeError) as e:
             # Propagate the specific error (ValueError for radius, TypeError for non-finite/type)
             raise e

    def createConicGradient(self, startAngle: float, x: float, y: float) -> ConicGradient:
        """Creates a ConicGradient object."""
        try:
             # Spec: Throw TypeError if arguments non-finite. Angle is in radians.
             params = [startAngle, x, y]
             if not all(math.isfinite(float(v)) for v in params):
                 raise TypeError("Gradient angle and coordinates must be finite numbers.")
             # Create the object
             return ConicGradient(startAngle, x, y)
        except (ValueError, TypeError) as e:
             # Reraise TypeError for spec compliance
             raise TypeError(f"Invalid arguments for createConicGradient: {e}") from e


    # --- Non-standard Kivy Helper Methods ---
    def clear_canvas(self):
         """Utility to completely clear all Kivy instructions from this widget's canvas
            and reset basic setup."""
         self._setup_canvas() # Clears canvas and adds UpdateNormalMatrix

    def get_size(self) -> tuple[float, float]:
         """Utility to get widget size as (width, height)."""
         return self.size

import math
class myapp(App):
    def build(self):
        ctx = Canvas2DContext()
        ctx.regFont('serif', 'Test/font.ttf')

        def test_draw(dt):
            ctx.clear_canvas()
            ctx.font = "48px serif"

            pass
            
        Clock.schedule_interval(test_draw, 0)
        return ctx

myapp().run()