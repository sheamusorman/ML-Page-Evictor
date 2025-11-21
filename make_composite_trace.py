import traces

def make_composite_trace():
  trace = []

  # --- Bucket A: Sequential scan ---
  tA = traces.get_trace("scan_seq")[:800]      # 800 accesses

  # --- Bucket C: Hotset (small WS) ---
  tC = traces.get_trace("hotset_small")[:800]

  # --- Bucket D: Loop pattern ---
  tD = traces.get_trace("loop_large")[:800]

  # --- Bucket E: Locality shift pattern ---
  tE = traces.get_trace("locality_shift")[:800]

  # --- Bucket B: Random access ---
  tB = traces.get_trace("random_pure")[:800]

  # --- Bucket F: Hard mixed region ---
  tF = []
  tsA = traces.get_trace("scan_seq")
  tsC = traces.get_trace("hotset_small")
  for i in range(400):
    if i % 2 == 0:
      tF.append(tsA[i])
    else:
      tF.append(tsC[i])

  trace = tA + tC + tD + tE + tB + tF
  return trace

# Write to file so the simulator can load it as a custom trace
if __name__ == "__main__":
  comp = make_composite_trace()
  with open("composite_trace.txt", "w") as f:
    for x in comp:
      f.write(str(x) + "\n")

  print("Composite trace written to composite_trace.txt")
