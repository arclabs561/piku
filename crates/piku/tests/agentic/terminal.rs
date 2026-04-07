use super::types::*;

pub struct TerminalObserver {
    parser: vt100::Parser,
}

impl TerminalObserver {
    pub fn new(rows: u16, cols: u16) -> Self {
        Self {
            parser: vt100::Parser::new(rows, cols, 500),
        }
    }

    pub fn process(&mut self, bytes: &[u8]) {
        self.parser.process(bytes);
    }

    pub fn snapshot(&self) -> ScreenSnapshot {
        let screen = self.parser.screen();
        let (term_rows, term_cols) = screen.size();

        let mut rows = Vec::with_capacity(term_rows as usize);
        for r in 0..term_rows {
            let mut row = String::new();
            for c in 0..term_cols {
                if let Some(cell) = screen.cell(r, c) {
                    row.push_str(cell.contents());
                }
            }
            rows.push(row.trim_end().to_string());
        }

        let interesting_rows = [term_rows.saturating_sub(1), term_rows.saturating_sub(2)];
        let styled_rows = interesting_rows
            .iter()
            .map(|&r| self.extract_styled_row(&screen, r, term_cols))
            .collect();

        ScreenSnapshot {
            contents: screen.contents(),
            rows,
            cursor: screen.cursor_position(),
            cursor_visible: !screen.hide_cursor(),
            styled_rows,
            size: (term_rows, term_cols),
        }
    }

    fn extract_styled_row(&self, screen: &vt100::Screen, row: u16, cols: u16) -> StyledRow {
        let mut cells = Vec::new();
        let mut text = String::new();
        for c in 0..cols {
            if let Some(cell) = screen.cell(row, c) {
                let ch = cell.contents().to_string();
                text.push_str(&ch);
                cells.push(StyledCell {
                    ch,
                    bold: cell.bold(),
                    dim: cell.dim(),
                    italic: cell.italic(),
                    inverse: cell.inverse(),
                    fg: cell.fgcolor().into(),
                    bg: cell.bgcolor().into(),
                });
            }
        }
        StyledRow {
            row_index: row,
            cells,
            text: text.trim_end().to_string(),
        }
    }
}
