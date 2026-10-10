# leed_daylight/deliver/calculator/xlsx_rows.py
"""A workbook edited as XML inside its zip (calculator spec 5.1, D5).

``Book`` holds every part as bytes; only the parts an edit touches are
rewritten, so a macro project, form controls, styles and protection come
back byte for byte. Parts are edited with regular expressions on their
text, never parsed and serialised again: a serialiser renames namespace
prefixes, and Excel then refuses the file.
"""
from __future__ import annotations

import dataclasses
import html
import pathlib
import posixpath
import re
import zipfile
from collections.abc import Mapping

from .xlsx_refs import col_number, shift_formula, shift_sqref, translate_formula

__all__ = ["Book", "Cell", "force_recalculation", "insert_rows", "set_cells", "unshare"]


@dataclasses.dataclass(frozen=True)
class Cell:
    """A cell's formula (without ``=``) and its value as text; None when absent."""
    formula: str | None
    value: str | None


_CELL = r'<c r="{ref}"(?P<attrs>[^>]*?)(?:/>|>(?P<body>.*?)</c>)'
#: An opening ``<f>`` tag, with or without attributes (``ca="1"``, ``t="array"``).
_F_OPEN = r"<f(?:\s[^>]*)?(?<!/)>"


class Book:
    def __init__(self, infos: list[zipfile.ZipInfo], parts: dict[str, bytes]) -> None:
        self._infos = infos
        self.parts = parts

    @classmethod
    def read(cls, path: str | pathlib.Path) -> Book:
        with zipfile.ZipFile(path) as z:
            infos = z.infolist()
            return cls(infos, {i.filename: z.read(i.filename) for i in infos})

    def write(self, path: str | pathlib.Path) -> None:
        """Every part still held, in the template's order and with its own
        compression; a removed part is left out."""
        with zipfile.ZipFile(path, "w") as z:
            for info in self._infos:
                if info.filename in self.parts:
                    z.writestr(info, self.parts[info.filename], compress_type=info.compress_type)

    def text(self, part: str) -> str:
        return self.parts[part].decode("utf-8")

    def put(self, part: str, text: str) -> None:
        self.parts[part] = text.encode("utf-8")

    def rels(self, part: str) -> dict[str, str]:
        """Relationship id -> target part, for ``part``; external targets left out."""
        folder, base = posixpath.split(part)
        name = f"{folder}/_rels/{base}.rels"
        if name not in self.parts:
            return {}
        found = {}
        for m in re.finditer(r"<Relationship ([^>]*?)/?>", self.text(name)):
            attrs = dict(re.findall(r'(\w+)="([^"]*)"', m.group(1)))
            if attrs.get("TargetMode") != "External" and "Target" in attrs:
                found[attrs["Id"]] = posixpath.normpath(posixpath.join(folder, attrs["Target"]))
        return found

    def sheets(self) -> dict[str, str]:
        """Sheet name -> its part, in workbook order."""
        rels = self.rels("xl/workbook.xml")
        return {html.unescape(name): rels[rid] for name, rid in
                re.findall(r'<sheet name="([^"]*)"[^>]*?r:id="([^"]*)"', self.text("xl/workbook.xml"))}

    def _shared(self) -> list[str]:
        if "xl/sharedStrings.xml" not in self.parts:
            return []
        items = re.findall(r"<si>(.*?)</si>", self.text("xl/sharedStrings.xml"), re.S)
        return [html.unescape("".join(re.findall(r"<t[^>]*>(.*?)</t>", s, re.S))) for s in items]

    def cell(self, sheet: str, ref: str) -> Cell:
        xml = unshare(self.text(self.sheets()[sheet]))
        m = re.search(_CELL.format(ref=ref), xml, re.S)
        if m is None:
            return Cell(None, None)
        attrs, body = m.group("attrs"), m.group("body") or ""
        f = re.search(_F_OPEN + r"(.*?)</f>", body, re.S)
        formula = html.unescape(f.group(1)) if f else None
        if 't="inlineStr"' in attrs:
            value = html.unescape("".join(re.findall(r"<t[^>]*>(.*?)</t>", body, re.S)))
            return Cell(formula, value)
        v = re.search(r"<v>(.*?)</v>", body, re.S)
        if v is None:
            return Cell(formula, None)
        value = html.unescape(v.group(1))
        if 't="s"' in attrs:
            value = self._shared()[int(value)]
        return Cell(formula, value)


_SHARED_ATTR = re.compile(r'\s+(?:t|ref|si)="[^"]*"')
_FORMULA_TAG = re.compile(r'(<c r="([A-Z]+)(\d+)"[^>]*>)\s*<f((?:\s[^>]*?)?)(/>|>)')


def _f_attrs(text: str) -> dict[str, str]:
    return dict(re.findall(r'(\w+)="([^"]*)"', text))


def unshare(xml: str) -> str:
    """Every shared formula of a sheet written out in each of its cells, so
    each cell's formula can be read and moved on its own. The shared
    attributes go; any other attribute of the formula (``ca``) stays. A
    child whose master is missing is refused, naming the cell."""
    masters: dict[str, tuple[str, int, str]] = {}
    for m in re.finditer(r'<c r="([A-Z]+)(\d+)"[^>]*>\s*<f((?:\s[^>]*?)?)(?<!/)>(.*?)</f>', xml, re.S):
        attrs = _f_attrs(m.group(3))
        if attrs.get("t") == "shared" and "ref" in attrs and "si" in attrs:
            masters[attrs["si"]] = (m.group(1), int(m.group(2)), html.unescape(m.group(4)))

    def rewrite(m: re.Match) -> str:
        open_tag, col, row, attrs, end = m.group(1), m.group(2), int(m.group(3)), m.group(4), m.group(5)
        fa = _f_attrs(attrs)
        if fa.get("t") != "shared":
            return m.group(0)
        keep = _SHARED_ATTR.sub("", attrs)
        if end == ">":
            return f"{open_tag}<f{keep}>"
        si = fa.get("si")
        if si not in masters:
            raise ValueError(f"{col}{row}: shared formula {si} has no master cell")
        mcol, mrow, text = masters[si]
        moved = translate_formula(text, row - mrow, col_number(col) - col_number(mcol))
        return f"{open_tag}<f{keep}>{html.escape(moved, quote=False)}</f>"

    return _FORMULA_TAG.sub(rewrite, xml)


#: The control characters XML 1.0 does not allow (tab, line feed and carriage return are allowed).
_ILLEGAL = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _cell_xml(ref: str, attrs: str, value: str | float | int) -> str:
    """A cell holding ``value``: text as an inline string (escaped, never a
    formula; XML-illegal control characters dropped, ``_x`` written as
    ``_x005F_x`` so Excel never decodes it), a number as a number (``5``
    not ``5.0``)."""
    attrs = re.sub(r'\s+t="[^"]*"', "", attrs)
    if isinstance(value, str):
        # XML 1.0 has no place for these controls; Excel reads "_xHHHH_" as an escaped character.
        text = html.escape(_ILLEGAL.sub("", value).replace("_x", "_x005F_x"), quote=False)
        return f'<c r="{ref}"{attrs} t="inlineStr"><is><t xml:space="preserve">{text}</t></is></c>'
    number = float(value)
    shown = str(int(number)) if number.is_integer() else repr(number)
    return f'<c r="{ref}"{attrs}><v>{shown}</v></c>'


def set_cells(book: Book, sheet: str, values: Mapping[str, str | float | int]) -> None:
    """Write each value: text as an inline string (never read as a formula
    or markup), a number as a number. The cell keeps its style. A missing
    cell is added in column order; a cell holding a formula is refused."""
    part = book.sheets()[sheet]
    xml = unshare(book.text(part))
    for ref, value in values.items():
        m = re.search(_CELL.format(ref=ref), xml, re.S)
        if m is not None:
            if re.search(r"<f[\s/>]", m.group("body") or ""):
                raise ValueError(f"{sheet}!{ref} holds a formula; the calculator map is wrong")
            xml = xml[:m.start()] + _cell_xml(ref, m.group("attrs"), value) + xml[m.end():]
            continue
        col, row = re.fullmatch(r"([A-Z]+)(\d+)", ref).groups()
        r = re.search(rf'<row r="{row}"[^>]*?(?:/>|>(.*?)</row>)', xml, re.S)
        if r is None:
            raise ValueError(f"{sheet}!{ref}: row {row} is not in the sheet")
        new = _cell_xml(ref, "", value)
        if r.group(0).endswith("/>"):
            xml = xml[:r.start()] + r.group(0)[:-2] + ">" + new + "</row>" + xml[r.end():]
            continue
        body = r.group(1) or ""
        later = [c for c in re.finditer(r'<c r="([A-Z]+)\d+"', body) if col_number(c.group(1)) > col_number(col)]
        at = r.start(1) + (later[0].start() if later else len(body))
        xml = xml[:at] + new + xml[at:]
    book.put(part, xml)


def force_recalculation(book: Book) -> None:
    """Excel and LibreOffice compute every formula again on open: cached
    formula values dropped, the calculation chain removed, full calculation
    on load set."""
    for part in book.sheets().values():
        def drop(m: re.Match) -> str:
            attrs = re.sub(r'\s+t="(?:str|b|e|n)"', "", m.group(1))
            return f"<c{attrs}>{m.group(2)}</c>"
        # The formula and the value are plain text (escaped), so [^<]* keeps the match inside one cell.
        book.put(part, re.sub(rf"<c( [^>]*)>({_F_OPEN}[^<]*</f>)<v>[^<]*</v></c>", drop, unshare(book.text(part))))
    book.parts.pop("xl/calcChain.xml", None)
    book.put("[Content_Types].xml",
             re.sub(r'<Override PartName="/xl/calcChain.xml"[^>]*/>', "", book.text("[Content_Types].xml")))
    book.put("xl/_rels/workbook.xml.rels",
             re.sub(r'<Relationship [^>]*Target="calcChain.xml"[^>]*/>', "", book.text("xl/_rels/workbook.xml.rels")))
    workbook = book.text("xl/workbook.xml")
    if re.search(r"<calcPr\b", workbook):
        workbook = re.sub(
            r"<calcPr([^>]*?)/>",
            lambda m: "<calcPr" + re.sub(r'\s+fullCalcOnLoad="[^"]*"', "", m.group(1)) + ' fullCalcOnLoad="1"/>',
            workbook)
    else:
        # The schema's order: calcPr comes after definedNames, else after sheets, and before anything later.
        anchor = "</definedNames>" if "</definedNames>" in workbook else "</sheets>"
        workbook = workbook.replace(anchor, anchor + '<calcPr fullCalcOnLoad="1"/>', 1)
    book.put("xl/workbook.xml", workbook)


def _formulas(xml: str, fn) -> str:
    """Every formula text of a sheet through ``fn``; the tag's attributes stay."""
    return re.sub(rf"({_F_OPEN})(.*?)</f>",
                  lambda m: m.group(1) + html.escape(fn(html.unescape(m.group(2))), quote=False) + "</f>",
                  xml, flags=re.S)


def _copy_row(row_xml: str, src: int, dst: int) -> str:
    """Row ``src`` as row ``dst``: styles kept, formulas moved, values
    dropped (``AddA`` fills the row, then clears its constants)."""
    out = re.sub(r'<row r="\d+"', f'<row r="{dst}"', row_xml, count=1)

    def cell(m: re.Match) -> str:
        col, attrs, body = m.group(1), re.sub(r'\s+t="[^"]*"', "", m.group(2)), m.group(3) or ""
        f = re.search(rf"({_F_OPEN})(.*?)</f>", body, re.S)
        if f:
            moved = translate_formula(html.unescape(f.group(2)), dst - src, 0)
            # An array formula's own range moves with the copy.
            tag = re.sub(r'(\sref=")([^"]*)"',
                         lambda r: r.group(1) + re.sub(r"(\$?[A-Z]+\$?)(\d+)", lambda n: f"{n.group(1)}{int(n.group(2)) + dst - src}", r.group(2)) + '"',
                         f.group(1))
            return f'<c r="{col}{dst}"{attrs}>{tag}{html.escape(moved, quote=False)}</f></c>'
        return f'<c r="{col}{dst}"{attrs}/>'

    return re.sub(r'<c r="([A-Z]+)\d+"([^>]*?)(?:/>|>(.*?)</c>)', cell, out, flags=re.S)


#: One whole row: an empty ``<row .../>`` or ``<row ...>...</row>``. The
#: attributes never hold ``>``, so ``[^>]*`` cannot run into the cells.
_ROW = r'<row r="(\d+)"([^>]*?/>|[^>]*>.*?</row>)'


def _shift_rows(xml: str, after_row: int, count: int) -> str:
    def row(m: re.Match) -> str:
        r = int(m.group(1))
        if r <= after_row:
            return m.group(0)
        rest = re.sub(r'(<c r="[A-Z]+)(\d+)"', lambda c: f'{c.group(1)}{int(c.group(2)) + count}"', m.group(2))
        return f'<row r="{r + count}"{rest}'
    return re.sub(_ROW, row, xml, flags=re.S)


def _shift_anchors(xml: str, after_row: int, count: int) -> str:
    """Zero-based anchor rows (drawings, form controls, VML): an index at or
    after ``after_row`` is a 1-based row below the insert."""
    def bump(n: int) -> int:
        return n + count if n >= after_row else n

    xml = re.sub(r"<xdr:row>(\d+)</xdr:row>", lambda m: f"<xdr:row>{bump(int(m.group(1)))}</xdr:row>", xml)

    def anchor(m: re.Match) -> str:
        values = [v.strip() for v in m.group(1).split(",")]
        values[2], values[6] = str(bump(int(values[2]))), str(bump(int(values[6])))
        return "<x:Anchor>" + ", ".join(values) + "</x:Anchor>"

    xml = re.sub(r"<x:Anchor>(.*?)</x:Anchor>", anchor, xml, flags=re.S)
    return re.sub(r"<x:Row>(\d+)</x:Row>", lambda m: f"<x:Row>{bump(int(m.group(1)))}</x:Row>", xml)


def insert_rows(book: Book, sheet: str, after_row: int, count: int) -> None:
    """Insert ``count`` copies of row ``after_row`` of ``sheet`` below it,
    as the calculator's ``AddA`` macro does (calculator spec 5.2). Every
    formula of every sheet, the defined names, and the sheet's merged
    cells, validation and conditional-format ranges, dimension, drawing,
    control and comment anchors move with their rows."""
    if count <= 0:
        return
    parts = book.sheets()
    target = parts[sheet]
    if not any(int(m.group(1)) == after_row for m in re.finditer(_ROW, book.text(target), re.S)):
        raise ValueError(f"{sheet}: row {after_row} is not in the sheet")
    for name, part in parts.items():
        xml = _formulas(unshare(book.text(part)), lambda f, own=name: shift_formula(f, own, sheet, after_row, count))
        if part == target:
            xml = _shift_rows(xml, after_row, count)
            # An array formula's own range moves with its rows.
            xml = re.sub(r'(<f\s[^>]*?\bref=")([^"]*)"',
                         lambda m: m.group(1) + shift_sqref(m.group(2), after_row, count) + '"', xml)
            found = next((m for m in re.finditer(_ROW, xml, re.S) if int(m.group(1)) == after_row), None)
            if found is None:
                raise ValueError(f"{sheet}: row {after_row} is not in the sheet")
            source = found.group(0)
            copies = "".join(_copy_row(source, after_row, after_row + k) for k in range(1, count + 1))
            xml = xml.replace(source, source + copies, 1)
            for attr in ("ref", "sqref"):
                xml = re.sub(rf'(<(?:mergeCell|dimension|hyperlink|conditionalFormatting|dataValidation)\b[^>]*? {attr}=")([^"]*)"',
                             lambda m: m.group(1) + shift_sqref(m.group(2), after_row, count) + '"', xml)
            xml = re.sub(r"<xm:sqref>(.*?)</xm:sqref>",
                         lambda m: "<xm:sqref>" + shift_sqref(m.group(1), after_row, count) + "</xm:sqref>", xml)
            xml = _shift_anchors(xml, after_row, count)
        # Validation and conditional-format formulas of any sheet may name the target.
        for tag in ("formula", "formula1", "formula2", "xm:f"):
            xml = re.sub(rf"<{tag}>(.*?)</{tag}>",
                         lambda m, t=tag, own=name: f"<{t}>" + html.escape(shift_formula(html.unescape(m.group(1)), own, sheet, after_row, count), quote=False) + f"</{t}>",
                         xml, flags=re.S)
        book.put(part, xml)
    for part in book.rels(target).values():
        if part not in book.parts or not part.endswith((".xml", ".vml")) or "/ctrlProps/" in part:
            continue
        xml = _shift_anchors(book.text(part), after_row, count)
        xml = re.sub(r'<comment ref="([^"]*)"', lambda m: f'<comment ref="{shift_sqref(m.group(1), after_row, count)}"', xml)
        book.put(part, xml)
    book.put("xl/workbook.xml", re.sub(
        r"(<definedName [^>]*>)(.*?)(</definedName>)",
        lambda m: m.group(1) + html.escape(shift_formula(html.unescape(m.group(2)), "", sheet, after_row, count),
                                           quote=False) + m.group(3),
        book.text("xl/workbook.xml"), flags=re.S))
