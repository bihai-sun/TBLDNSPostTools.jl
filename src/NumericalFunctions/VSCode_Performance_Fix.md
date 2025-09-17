# VS Code Performance Issues - Diagnosis & Solutions

## Current Problem
VS Code is hanging due to high CPU usage (99%) and excessive memory consumption (1.8GB). The system load is extremely high (32.12).

## Immediate Solutions

### 1. Restart VS Code (Recommended)
```bash
# Force quit VS Code and all processes
pkill -f "Visual Studio Code"
pkill -f "Code Helper"
pkill -f "julia.*notebook"

# Wait 10 seconds, then restart VS Code
sleep 10
open -a "Visual Studio Code"
```

### 2. Clear VS Code Cache
```bash
# Close VS Code first, then:
rm -rf ~/Library/Application\ Support/Code/CachedData/*
rm -rf ~/Library/Application\ Support/Code/logs/*
rm -rf ~/Library/Application\ Support/Code/User/workspaceStorage/*
```

### 3. Disable Extensions Temporarily
- Open VS Code in safe mode: `code --disable-extensions`
- Or disable specific extensions that might cause issues:
  - Julia extension
  - Jupyter extension
  - Large file viewers

## Long-term Solutions

### 1. Notebook Optimization
- **Keep notebooks small**: Split large notebooks into smaller ones
- **Clear outputs regularly**: `Kernel → Restart & Clear Outputs`
- **Close unused tabs**: Don't keep many large files open
- **Use external plotting**: Save plots to files instead of inline display

### 2. VS Code Settings
Add to your `settings.json`:
```json
{
    "notebook.output.textLineLimit": 30,
    "notebook.output.scrolling": true,
    "notebook.cell.outputScrolling": true,
    "jupyter.outputScrolling": true,
    "files.watcherExclude": {
        "**/gr-temp/**": true,
        "**/.julia/**": true
    }
}
```

### 3. System Resources
- **Close other applications**: Free up memory and CPU
- **Monitor file sizes**: Keep notebooks under 500KB
- **Use Activity Monitor**: Check for runaway processes

## Prevention Tips

1. **Regular maintenance**:
   - Restart VS Code daily if working with large notebooks
   - Clear outputs after each session
   - Keep workspace clean

2. **File management**:
   - Move large output files out of workspace
   - Use `.gitignore` for temporary files
   - Clean up `gr-temp/` directory regularly

3. **Performance monitoring**:
   - Check file sizes: `du -h *.ipynb`
   - Monitor processes: `top -o cpu`
   - Watch memory usage in Activity Monitor

## Current Status
- Notebook size: 72KB ✅ (Good)
- Number of lines: 1700 ⚠️ (Long but manageable)
- System load: 32.12 ❌ (Critical - needs restart)
- VS Code CPU: 99% ❌ (Critical - needs restart)

**Recommendation**: Force restart VS Code immediately using the commands above.
