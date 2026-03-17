import json

filepath = "c:/Users/YASH-SHIVGAN/Desktop/Crypto/telegram_engine/testing/testing.ipynb"

try:
    with open(filepath, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # find the execution cell
    execution_cell_idx = -1
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code' and len(cell['source']) > 0 and "#               EXECUTION                    #" in cell['source'][1]:
            execution_cell_idx = i
            break
            
    if execution_cell_idx != -1:
        source = nb['cells'][execution_cell_idx]['source']
        
        # We want to find the line that assigns 'filtered_trends' and add the sort right after
        new_source = []
        for line in source:
            new_source.append(line)
            if "or (t[\"side\"] == \"low\" and t[\"direction\"] == \"up\")" in line:
                # the next line is "    ]\n"
                pass
            if line.strip() == "]" and "or (t[\"side\"] == \"low\"" in new_source[-2]:
                new_source.append("    \n")
                new_source.append("    # Sort chronologically by the completion of the pattern (pivot 3)\n")
                new_source.append("    filtered_trends.sort(key=lambda x: x[\"pivot_3_idx\"])\n")
                print("Injected sort logic.")
        
        nb['cells'][execution_cell_idx]['source'] = new_source
        nb['cells'][execution_cell_idx]['outputs'] = []
        nb['cells'][execution_cell_idx]['execution_count'] = None
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1)
        print("Successfully updated the notebook.")
    else:
        print("Execution cell not found.")
except Exception as e:
    import traceback
    traceback.print_exc()
