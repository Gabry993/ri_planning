import lxml.etree as ET
from IPython.display import HTML, DisplayHandle, display
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import XmlLexer


def xml(xml_element: ET.Element) -> DisplayHandle:
    formatter = HtmlFormatter()
    xml_indented = ET.tostring(
        xml_element, pretty_print=True, encoding='unicode')
    style = formatter.get_style_defs('.highlight')
    code = highlight(xml_indented, XmlLexer(), formatter)
    return display(HTML(f'<style type="text/css">{style}</style>    {code}'))
