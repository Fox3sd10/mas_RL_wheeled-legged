import re
path = "/home/brown/wheeled-legged_RL/COD-2026RoboMaster-Balance-Simulation_File/fix_cod_usd_naming.py"
src = open(path).read()

old_block = '''    # 6) 重定向所有 Relationship 中的旧路径 -> 新路径
    n_rel = 0
    for prim in stage.Traverse():
        for prop in prim.GetProperties():
            if prop.GetPropertyStack and False:
                pass
            if isinstance(prop, Usd.Relationship):
                rel = Usd.Relationship(prop)
                targets = rel.GetTargets()
                changed = False
                new_targets = []
                for t in targets:
                    s = str(t)
                    if s in path_map:
                        new_targets.append(path_map[s])
                        changed = True
                    else:
                        new_targets.append(t)
                if changed:
                    rel.SetTargets(new_targets)
                    n_rel += 1
    print(f"[OK] 已重定向 {n_rel} 个 Relationship")
'''

new_block = '''    # 6) 重定向所有 Relationship 中的旧路径 -> 新路径
    #    使用 prim.GetRelationships() 获取该 prim 上的所有关系关系。
    n_rel = 0
    for prim in stage.Traverse():
        for rel in prim.GetRelationships():
            targets = rel.GetTargets()
            changed = False
            new_targets = []
            for t in targets:
                s = str(t)
                if s in path_map:
                    new_targets.append(path_map[s])
                    changed = True
                else:
                    new_targets.append(t)
            if changed:
                rel.SetTargets(new_targets)
                n_rel += 1
    print(f"[OK] 已重定向 {n_rel} 个 Relationship")
'''

assert old_block in src, "old_block not found!"
src = src.replace(old_block, new_block)
open(path, "w").write(src)
print("patched OK")
