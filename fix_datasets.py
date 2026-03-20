import json
import glob

files = glob.glob('eval/datasets/constraint_tracking/llm_generated/*.json')

for f in files:
    try:
        with open(f, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        d = data[0] if isinstance(data, list) else data
        modified = False
        
        if 'domain' not in d:
            if 'software_engineering' in f: d['domain'] = 'Software Engineering'
            elif 'js_utils' in f: d['domain'] = 'Software Engineering'
            elif 'python_backend' in f: d['domain'] = 'Software Engineering'
            elif 'sql_databases' in f: d['domain'] = 'Data Science'
            elif 'cooking' in f: d['domain'] = 'Cooking Recipes'
            elif 'finance' in f: d['domain'] = 'Personal Finance'
            else: d['domain'] = 'General'
            modified = True
            
        for cp in d.get('checkpoints', []):
            if 'after_turn_id' in cp:
                cp['turn'] = cp.pop('after_turn_id')
                modified = True
            if 'constraint' in cp:
                cp['constraint_tested'] = cp.pop('constraint')
                modified = True
            if 'checkpoint_id' in cp:
                del cp['checkpoint_id']
                modified = True
                
        if modified:
            with open(f, 'w', encoding='utf-8') as file:
                json.dump([d] if isinstance(data, list) else d, file, indent=2)
            print("Fixed file: " + f)
    except Exception as e:
        print("Error fixing file " + f + ": " + str(e))

print("Keys fixed successfully across all generating datasets!")
