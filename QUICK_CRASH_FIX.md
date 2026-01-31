# 🚨 QUICK CRASH FIX REFERENCE

## 🔴 If Kernel Crashes RIGHT NOW

### Immediate Actions:
1. **Runtime** → **Restart runtime**
2. **Edit** → **Clear all outputs**
3. Re-run cells **ONE AT A TIME**
4. Wait **5 seconds between cells**

---

## ✅ Key Changes in Fixed Notebook

### 1. Memory Management (NEW!)
- Automatic garbage collection
- GPU cache clearing
- Thread limiting

### 2. Safer Defaults
- Episodes: 100 → **50**
- Batch size: 128 → **64**
- DPI: 300 → **150**

### 3. Error Handling
- Training won't crash on single error
- Continues to next episode automatically

### 4. Cleanup Cells (NEW!)
- After Q-Learning
- After DQN
- After DDQN

---

## 🎯 How to Run Without Crashes

### Step-by-Step:
```
1. Upload: DRL_Assignment2_Complete_CrashFixed.ipynb
2. Runtime → Change runtime type → GPU
3. Run cells ONE AT A TIME
4. Wait 5 seconds between cells
5. Watch for "✓ Memory cleaned up" messages
```

### Critical Points to Restart:
- ✅ After Q-Learning completes
- ✅ Before DQN training starts
- ✅ Before final comparison

---

## ⚡ Emergency Fixes

### If Still Crashes:

**Fix 1** - Reduce Episodes:
```python
MAX_EPISODES = 25  # In Config cell
```

**Fix 2** - Reduce Batch Size:
```python
DQN_BATCH_SIZE = 32  # In Config cell
```

**Fix 3** - Use CPU Only:
```python
device = 'cpu'  # In DQN training cell
```

**Fix 4** - Skip Visualizations:
Comment out `plt.savefig()` and `plt.show()` lines

---

## 📊 Safe Episode Counts

| Episodes | Time | Risk |
|----------|------|------|
| 25 | 5 min | ✅ Safe |
| 50 | 10 min | ✅ Safe |
| 100 | 20 min | ⚠️ Restart between sections |
| 1000+ | Hours | 🔴 Needs multiple restarts |

---

## ⚠️ Warning Signs

**About to crash if you see:**
- RAM > 90%
- GPU Memory > 14GB
- "ResourceExhausted" errors
- Notebook becoming slow

**Action**: Restart immediately!

---

## ✅ Success Checklist

- [ ] Using crash-fixed notebook
- [ ] GPU enabled
- [ ] Running cells one at a time
- [ ] Waiting between cells
- [ ] Seeing "✓ Memory" messages
- [ ] Restarting between sections

---

## 💡 Pro Tips

1. **Start Small**: Test with 25 episodes first
2. **Restart Often**: Don't trust long runs
3. **Download Results**: Save after each section
4. **Monitor Memory**: Watch GPU usage
5. **Be Patient**: Wait between cells

---

## 🆘 Still Crashing?

**Last Resort Options:**
1. Run in 3 separate notebooks (Q-Learning, DQN, DDQN)
2. Use Colab Pro (more memory)
3. Use local Jupyter (more control)
4. Reduce to 10-25 episodes only

---

## 📁 File You Need

**DRL_Assignment2_Complete_CrashFixed.ipynb**
← Use this one!

(Not the original - it will crash!)

---

## 🎯 Bottom Line

The fixed notebook has:
- ✅ Auto memory management
- ✅ Crash prevention
- ✅ Error handling
- ✅ Smaller defaults

**Just run cells slowly and restart between sections!**

---

**Need more details? Read CRASH_FIX_GUIDE.md**
