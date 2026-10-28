## Where to look (read only what the task needs)

| folder | content | details |
|---|---|---|
| `fem/` | FE framework (System, elements, mesh, Abaqus interface) | `fem/CLAUDE.md` |
| `materials/` | constitutive laws, one file each | — |
| `cmm/` | G&R models (hcmm, fcmm), wall, studies, results | `cmm/CLAUDE.md` |
| `verification/` | FE verification scripts and logs | `verification/CLAUDE.md` |
| `docs/` | HTML docs (FE design, usage, verification, glossary, literature) | `docs/CLAUDE.md` |
| `.context/` | papers, Maes UMATs, reference data (not in git) | `.context/CLAUDE.md` |

- Never list or grep recursively in `cmm/results/`, `cmm/compiled/` or the data folders of `.context/` (GBs of .vtu/.odb/.npz); open a single file by name.
- Open a paper only when the task is about it; `.context/CLAUDE.md` says which folder.
