## Launching the App (macOS)
If Finder reports a permission error:
1) Right-click `run_app.command` → Open (once), or
2) Run in Terminal:
   chmod +x run_app.command
   ./run_app.command

This launcher self-fixes permissions and starts the browser UI.


## If macOS says you don't have access privileges
If double‑clicking `run_app.command` shows a permissions error, run this once in Terminal from the project folder:

```bash
chmod +x run_app.command run_in_terminal.sh
./run_app.command
```

You can also run:

```bash
bash run_in_terminal.sh
```
