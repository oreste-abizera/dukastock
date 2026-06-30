# Diagrams

UML and architecture diagrams referenced in the proposal (Chapter 3.3 and
3.4) as Figures 2, 4, 5, 6, and 7. The original proposal described these
in prose only; this folder adds the actual rendered diagrams.

| File | Proposal reference | Content |
|---|---|---|
| `figure2_architecture.svg` / `.png` | Figure 2 | System Architecture Overview |
| `figure4_use_case.svg` / `.png` | Figure 4 | Use Case Diagram |
| `figure5_class_diagram.svg` / `.png` | Figure 5 | Class Diagram |
| `figure6_er_diagram.svg` / `.png` | Figure 6 | Entity-Relationship Diagram |
| `figure7_sequence_diagram.svg` / `.png` | Figure 7 | Sequence Diagram (WhatsApp sales logging + restock advice) |

## Using these in the thesis document

The `.png` files (1600px wide) are sized for direct embedding into the
Word document — insert them at the figure caption locations already
described in the proposal text (e.g. "*Figure 2: System Architecture
Overview*"). The `.svg` source files are kept alongside them so the
diagrams can be edited (text, colors, layout) without starting over —
any SVG editor or even a text editor works, since these are hand-authored
SVG, not exported from a GUI tool.

## Validation

Every SVG in this folder was rendered with `cairosvg` during the project
build to confirm it produces a valid, correctly laid-out image before
being included — see the build history for the specific bugs this caught
(an invalid XML comment containing `--`, and two label-overlap layout
issues in the ER diagram). If you edit these files, re-validate with:

```bash
pip install cairosvg
python3 -c "import cairosvg; cairosvg.svg2png(url='figure2_architecture.svg', write_to='/tmp/check.png', output_width=1600)"
```

then open `/tmp/check.png` to confirm the edit didn't break anything.

## Noted deviations from the literal proposal text

- **Figure 5 (Class Diagram)**: `ShopkeeperProfile` includes a `created_at`
  field not in the original proposal's class diagram text — a standard
  audit timestamp added during implementation, flagged with a footnote
  directly on the diagram. See `backend/app/models/orm.py`.
- All other attributes, relationships, and cardinalities match the
  proposal's textual description exactly.
