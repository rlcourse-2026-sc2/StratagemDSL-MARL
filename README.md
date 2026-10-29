# StratagemDSL-MARL：用 sc2te_env 训练 StratagemDSL 生成的地图

[rlcourse-2026-sc2](https://github.com/rlcourse-2026-sc2) 的训练类下游仓库：

| 仓库 | 作用 |
| --- | --- |
| [StratagemDSL-RL](https://github.com/rlcourse-2026-sc2/StratagemDSL-RL) | 上游：用 `.strat` 描述计策，生成 `.SC2Map` |
| [LLM-PySC2](https://github.com/rlcourse-2026-sc2/LLM-PySC2) | 下游：LLM 智能体在生成的地图上对战 |
| **StratagemDSL-MARL**（本仓库） | 下游：多智能体强化学习（sc2te_env / SMAC 接口，QMIX、WQMIX、MAPPO……）在生成的地图上训练 |

把 [StratagemDSL-RL](https://github.com/rlcourse-2026-sc2/StratagemDSL-RL) 用 `.strat` 生成的任何 `.SC2Map`，
直接变成 [sc2te_env](https://github.com/yungongWang/sc2te_env)（SMAC 风格的多智能体环境）里的一个训练任务，
不用再为每张图手写 `map_param_registry` 条目和 `star36env_xxx.py` 子类。

```text
StratagemDSL 这边（不需要 SC2）                         sc2te_env 这边（训练）
───────────────────────────────                         ─────────────────────────────────────────────
my.strat ──tools/strat2sc2te.py──▶ my.SC2Map           StratagemTacticsEnv(".../my.sc2te.json")
                                   my.sc2te.json  ───▶    · 自动注册地图（PySC2 + sc2te 注册表）
                                   （部队、区域、            · agents / enemies 槽位来自 .strat 的部队
                                    intent 胜负条件、         · 胜负 = .strat 的 intent success / failure
                                    单位数值、计划）          · 动作 = SMAC 动作 + 移动到命名区域 + 攻击敌方部队
                                                          · reset 利用地图自带的全灭重刷触发器
                                                          ──▶ PyMARL / pymarl2 / 自己的训练循环
```

## 0. sc2te_env 能帮上什么忙

两个仓库正好是一条流水线的两半：

| | StratagemDSL-RL | sc2te_env |
| --- | --- | --- |
| 做什么 | 用 DSL 描述计策，编译成**地图**：双方兵力、出生点、命名区域、敌方按计划（Petri 网）行动的 Galaxy 触发器 | 把一张 SC2 地图包装成 **SMAC 接口的多智能体 RL 环境**：每个己方单位一个智能体，`get_obs / get_state / get_avail_actions / step`，可直接接 PyMARL（QMIX、VDN、MAPPO……） |
| 己方单位谁来指挥 | 地图脚本**不**指挥己方（`agent_intents.json` 只是建议），留给智能体 | 智能体通过 raw 接口逐单位下命令 |
| 胜负 | 计划到达 `terminal success/failure`（MARL 地图只记录不结束；`--gameover` 地图调用 GameOver） | 每张图手写判定（`check_structure` 等） |

StratagemDSL 默认打出来的地图本来就是给 MARL 用的：每回合结束靠 “双方单位全灭 → `gt_Reset` 重新刷兵”，
这和 SMAC/sc2te 的重置方式（`_restart` 杀光所有单位）是同一套机制。但直接拿 sc2te 的 `SC2TacticsEnv` 去跑会遇到：

1. 每张新图都要在 `sc2_tactics_maps.py` 里手写 `n_agents / n_enemies / 种族 / unit_type_bits`，再写一个 `star36env_xxx.py` 处理单位类型编号、胜负判定；
2. sc2te 的胜负是“一方全灭”或手写的建筑判定，而计策图的胜负是 `.strat` 里的 `intent`（例如“摧毁敌方指挥中心 **或** 击溃守军，且己方主基地存活”）；
3. 己方建筑（主基地、虫洞）在 sc2te 里会被当成智能体；计划中途才刷出来的援军（`initial = false`）会让 `init_units` 一直等不到“全员到齐”；
4. 只杀“被跟踪的单位”不一定能触发重刷（例如虫族建筑死亡掉出的巢虫、援军），会卡在 reset。

本包就是补上这一层。

## 1. 包里有什么

```text
StratagemDSL-MARL/
├── tools/strat2sc2te.py          StratagemDSL 侧导出器：.strat → 打包 .SC2Map + <名字>.sc2te.json
├── sc2_tactics/stratagem/        sc2te 侧（装到 smac/env/sc2_tactics/stratagem）
│   ├── env.py                    StratagemTacticsEnv（SC2TacticsEnv 的子类）
│   ├── spec.py                   读取 .sc2te.json；单位名 → SC2 单位 ID（用 sc2te 自带的 constants/Units.json）
│   ├── tracker.py                单位 → 槽位 的匹配；intent 求值需要的世界事实
│   ├── conditions.py             intent / derived 表达式的三值求值（与 Galaxy 生成代码语义一致）
│   ├── maps.py                   运行时注册地图（PySC2 Map 类 + sc2te map_param_registry）
│   ├── plan.py                   （可选）用 StratagemDSL 的 PlanTracker 跟踪计划进度：阶段奖励、阶段观测
│   └── fake_sc2.py               （可选）不装 SC2 的协议级替身（StratagemDSL 的 PlanSim），用于调试流程
├── examples/
│   ├── quickstart.py             随机策略跑几局
│   ├── plan_follower.py          按计划行动的脚本基线（检验地图在这个动作空间里能不能赢）
│   └── pymarl/                   PyMARL 环境配置 sc2te_strat.yaml + 注册代码
├── tests/test_offline.py         离线测试（fake 引擎，跑全部示例计策）
└── install.py                    安装脚本（Linux / macOS / Windows 通用）
```

## 2. 安装

推荐**一个 Python 3.10 环境**装全部东西：PySC2 4.0 在 3.11+ 上有 `random.shuffle` 的兼容问题，StratagemDSL 要求 ≥3.10。
（导出器也可以放在 StratagemDSL 自己的环境里跑，训练机只要拿到 `.SC2Map` 和 `.sc2te.json` 两个文件即可，见 §3.1。）

```bash
conda create -n sc2te python=3.10 -y && conda activate sc2te

# ① SMAC（会带上 pysc2 4.0）；protobuf 太新会报 "Descriptors cannot be created directly"
pip install git+https://github.com/oxwhirl/smac.git "protobuf<=3.20.3"

# ② sc2te_env 是 smac/env 的覆盖层，③ 本包装到 smac/env/sc2_tactics/stratagem —— 一条命令完成：
git clone https://github.com/yungongWang/sc2te_env
python install.py --sc2te ./sc2te_env          # 已经覆盖过 sc2te_env 的话直接 python install.py
#   开发时可用 --link 改成软链接，改代码立即生效

# ④ StratagemDSL（导出器必需；训练时可选：计划跟踪 / fake 引擎要用）
git clone https://github.com/rlcourse-2026-sc2/StratagemDSL-RL
cd StratagemDSL-RL && pip install -r requirements.txt && pip install -e . && cd ..

# ⑤ 真机训练：StarCraft II（Linux 无头版 4.10 与 SMAC 相同），设置 SC2PATH
export SC2PATH=~/StarCraftII
```

装好后 `python -c "from smac.env.sc2_tactics.stratagem import StratagemTacticsEnv"` 不报错即可。

## 3. 使用

### 3.1 导出地图和场景描述（在 StratagemDSL-RL 根目录运行）

```bash
# 单个计策（底图不写时按文件名猜，和 strat pack 一样）
python ../StratagemDSL-MARL/tools/strat2sc2te.py examples/stratagems/dhls_18_enter_signal.strat \
    --base-map assets/base_maps/dhls_te.SC2Map -o build/sc2te

# 一次导出很多个（例如作业里自己写的 5 张图）
python ../StratagemDSL-MARL/tools/strat2sc2te.py work/*.strat -o build/sc2te

# 已经自己 strat pack 好的地图，只补写 .sc2te.json
python ../StratagemDSL-MARL/tools/strat2sc2te.py work/my_lure.strat \
    --map build/my_lure/my_lure.SC2Map --base-map "assets/base_maps/Agna Valley-1.SC2Map" -o build/sc2te
```

输出（每个计策一个子目录，打包产物和 `strat pack` 相同，另加一个 JSON）：

```text
[OK] dhls_18_enter_signal: zerg vs terran, n_agents=13 n_ally_assets=3 n_enemies=11 regions=6 limit=329
     map  build/sc2te/dhls_18_enter_signal/dhls_18_enter_signal.SC2Map
     spec build/sc2te/dhls_18_enter_signal/dhls_18_enter_signal.sc2te.json
```

常用参数：`--gameover`（一局即一回合，环境用 `RequestRestartGame` 重开，见 §5）、`--oracle`（地图自己执行己方计划，做对照组）、
`--limit N`（回合步数上限，默认按地图尺寸估算）、`--step-mul`、`--loose`、
`--svg`（顺便用 Graphviz 画 PlanGraph/PetriNet 的 SVG；默认不画，导出用不到，也避免 Graphviz 装坏时弹窗卡住）。

整个子目录可以拷到训练机上：`.SC2Map` 在 JSON 记录的绝对路径找不到时，会在 JSON 同目录下找。

### 3.2 在 sc2te_env 里跑

```python
from smac.env.sc2_tactics.stratagem import StratagemTacticsEnv

env = StratagemTacticsEnv("build/sc2te/dhls_18_enter_signal")   # 目录或 *.sc2te.json
print(env.describe())               # 双方部队、槽位数、胜负条件
info = env.get_env_info()           # n_agents / n_actions / obs_shape / state_shape / action_names ...

obs, state = env.reset()
reward, terminated, step_info = env.step([1] * env.n_agents)   # 1 = stop
print(env.last_detail)              # 本步事件（进/出区域、阵亡、刷出）、outcome、计划阶段
env.close()
```

命令行示例：

```bash
python examples/quickstart.py build/sc2te/dhls_18_enter_signal            # 随机策略，真机
python examples/plan_follower.py build/sc2te/dhls_18_enter_signal         # 按计划行动的脚本基线
# 没装 SC2 时加 --fake / --engine fake，用 StratagemDSL 的 PlanSim 代替游戏（仅用于调试流程）
python examples/plan_follower.py build/sc2te/dhls_18_enter_signal --engine fake --episodes 3
```

### 3.3 接 PyMARL / pymarl2

1. 在 `src/envs/__init__.py` 里注册（见 `examples/pymarl/envs_init_patch.py`）：
   ```python
   from smac.env.sc2_tactics.stratagem import StratagemTacticsEnv
   REGISTRY["sc2te_strat"] = partial(env_fn, env=StratagemTacticsEnv)
   ```
2. 把 `examples/pymarl/sc2te_strat.yaml` 复制到 `src/config/envs/`。
3. 训练：
   ```bash
   python src/main.py --config=qmix --env-config=sc2te_strat \
       with env_args.scenario=/abs/path/build/sc2te/dhls_18_enter_signal
   ```

按 sc2te 改过的 PyMARL 分支（例如 runner 里有 `reward, delta_enemy, delta_deaths, delta_ally = reward` 的 WQMIX 仓库）：

```bat
python src\main.py --config=qmix --env-config=sc2te_strat with env_args.scenario=E:/thesis/6.DSL/StratagemDSL-RL/build/sc2te/dhls_18_enter_signal env_args.reward_tuple=True evaluate=False checkpoint_path="" load_step=0
```

注意两点：yaml 里 `reward_tuple: False`（原版 PyMARL 要标量奖励；sc2te 自己的 `SC2TacticsEnv.step` 返回
`(reward, delta_enemy, delta_deaths, delta_ally)` 元组，本环境默认保持一致，`True`）；`info_detail: False`
（PyMARL 会把 info 的每个值相加，列表/字符串会报错；详细信息始终在 `env.last_detail` 里）。

## 4. 接口说明

### 4.1 `.sc2te.json`（`stratagemdsl.sc2te-scenario.v1`）

| 字段 | 内容 |
| --- | --- |
| `map` | 打包地图的绝对路径、文件名、底图、宽高、可玩区域、`gameover`、`oracle`、sha256 |
| `players` / `races` | 己方/敌方玩家号；种族（SMAC 字母 P/T/Z 和全名） |
| `forces` | 每个 force/asset：`side`、`owner`、`kind`、`composition`、`spawn`（世界坐标）、`initial`（开局刷出还是计划中途刷出）、`controlled`（= 己方 force，成为智能体） |
| `regions` | 命名区域：圆心、半径 |
| `intent` | `success / failure / primary / secondary` 的表达式 AST（与 `agent_intents.json` 同一格式），`text` 为可读文本 |
| `derived` | intent 引用的派生事实 |
| `unit_stats` | 用到的单位（含变形：SiegeTankSieged、RoachBurrowed……）的射程、视野、血量、护盾、是否空中/建筑、所属“家族” |
| `plan` | 阶段（id、目的、是否终局、己方命令）、转移（输入/输出、事件、条件） |
| `strategy_source` | .strat 原文（计划跟踪和 fake 引擎用，不依赖原文件） |
| `sizes` / `env` | 槽位数；建议的回合上限和 step_mul |

### 4.2 槽位：谁是智能体

| 组 | 来源 | 在哪 |
| --- | --- | --- |
| agents（`n_agents`） | 己方 **force** 的全部单位（含 `initial = false` 的后续部队，未刷出前视为“死亡”：只能 no-op） | 每个单位一个智能体 |
| ally assets（`n_ally_assets`） | 己方 **asset**（主基地、虫洞、水晶塔……） | obs 的友军块、state 的友军块里多出的行；不是智能体 |
| enemies（`n_enemies`） | 敌方全部 force 和 asset（含中途援军的空槽） | obs 敌人块、state 敌人块、攻击动作 |

单位第一次出现时按“同 owner、同单位家族、离出生点（或该部队当前中心）最近、该部队还有空槽”分配，之后按 tag 跟踪，
变形/钻地不会换槽。部队内按（单位类型、x、y）排序，所以同一部队的智能体编号在各回合稳定。
幻象、巢虫、蝗虫等不属于任何部队的单位不占槽位。

### 4.3 动作空间（`env.get_action_names()` 可打印）

```text
0            no-op（仅死亡/未刷出时可用）
1            stop
2..5         向北/南/东/西移动 move_amount 格（SMAC 原有）
+R           move_to:<区域>      移动到 .strat 的命名区域中心（region_action="attack" 时为攻击移动）   region_actions
+F           attack_force:<敌方部队>  攻击移动到该敌方 force/asset 当前中心（对应 DSL 的 attack(F, target)） force_actions
+Q           无目标技能（ability_actions 里 target=quick 的）
+E           attack:enemy_i     攻击第 i 个敌方槽位（在射程内才可用）
+…           有目标技能（target=enemy 每个敌人一个、target=ally 每个智能体一个）
```

区域/部队宏动作让智能体能直接表达“佯攻到 A 区域”“主力打对方基地”这类计策动作，不必靠一步 2 格的微操慢慢走。
攻击可用条件：`距离 ≤ max(武器射程 + 双方半径, min_shoot_range)`，`min_shoot_range=6` 与 SMAC 一致（近战单位会自己走过去）；
打不到空中/地面的单位对应动作不可用。

技能动作：`ability_actions=["burrow_down", "burrow_up", "siege", "unsiege", "heal", "neural_parasite", "stim", ...]`
（见 `env.py` 的 `ABILITY_PRESETS`），或自定义：

```python
StratagemTacticsEnv(spec, ability_actions=[
    "burrow_down", "burrow_up",
    {"name": "force_field", "ability_id": 1526, "target": "enemy", "unit_types": ["Sentry"], "range": 9, "energy": 50},
])
```

### 4.4 观测与状态

```text
obs  = move(4 [+8 寻路] [+9 高度])
     | 敌人   E × [可攻击, 距离, dx, dy, 血量, (护盾), 单位类型 one-hot]     （视野内才非零）
     | 友军 (A-1+K) × [可见, 距离, dx, dy, 血量, (护盾), 类型, (上一动作)]   （其他智能体 + 己方建筑）
     | 区域   R × [dx, dy, 是否在区域内]                                       obs_regions
     | 自身   [血量, (护盾), x/W, y/H, 类型, (时间步)]                          obs_own_position
     | (计划阶段 one-hot)                                                      obs_plan_phase
state = 友军 (A+K) × [血量, 冷却/能量, x, y, (护盾), 类型]
      | 敌人 E × [血量, x, y, (护盾), 类型]
      | (各智能体上一动作) | (时间步) | (计划阶段 one-hot)
```

单位类型 one-hot 用双方共享的“家族”表（`env.spec.families`，例如 SiegeTank 与 SiegeTankSieged 同一位），视野用单位真实视野
（`unit_stats.sight`）。有神族的一方才有护盾位。

### 4.5 奖励与终止

* 奖励：与 SMAC/sc2te 相同的伤害 + 击杀奖励（敌方**建筑**也计入，拆基地有回报）+ 胜利奖励，`reward_scale` 归一化；
  `reward_phase=x` 时，计划跟踪器每进入一个新的非终局阶段额外 +x（需要 StratagemDSL），用于鼓励“按计策走”。
* `termination="intent"`（默认）：每步在 raw 观测上求值 `.strat` 的 `intent.success / failure`，成立即胜/负；同一步都成立按失败
  （`failure_first`）。支持的函数：`alive destroyed defeated count count_in in_region distance nearest_distance killed_in
  targeting life_ratio construction_complete under_construction airborne player_unit_count mission_time after`，
  以及计划相关的 `elapsed tokens active`（需计划跟踪）。不支持的函数视为“未知”，永不触发结束（导出时会警告）。
* `termination="plan"`：跟踪到的计划到达终局阶段时结束；`"army"`：SMAC 规则（一方全灭）。
* `defeat_on_army_loss=True`：全部智能体阵亡且没有待刷出的己方部队时判负。
* 用 `--gameover` 打包的地图：游戏自己 GameOver，环境读 `player_result`。
* 超过 `episode_limit` 步：超时（`continuing_episode=True` 时 `info["episode_limit"]=True`）。
* `step` 的 info（数值，PyMARL 安全）：`battle_won`、`battle_lost`、`dead_allies`、`dead_enemies`；
  `env.last_detail`：`outcome`（success/failure/draw/timeout）、`events`（enter/leave/died/spawned）、`plan_phases`、`plan_fired`。

### 4.6 重置

* 普通（MARL）地图：杀掉双方**所有**单位（包括未跟踪的、巢虫等），等到出现足够多的“新 tag 且属于某个部队”的单位才算重刷完成，
  多次失败才整个重启 SC2。
* `--gameover` 地图：`RequestRestartGame` 重开对局（任务计时也随之归零）。
* 回合开始时等待开局部队全部出现（最多 40 个 game loop，数量稳定后也会继续并打印警告）。

### 4.7 全部构造参数

与 `SC2TacticsEnv` 相同的参数（`step_mul move_amount difficulty game_version seed continuing_episode obs_* state_* reward_*
replay_dir replay_prefix window_size_* debug`）之外：

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `scenario`（或 `spec`） | 必填 | `.sc2te.json` 路径、所在目录、dict 或 `ScenarioSpec` |
| `map_name` | `strat_<名字>` | 注册名（不能与 sc2te 内置图重名） |
| `step_mul` / `episode_limit` | 取自 JSON | |
| `difficulty` | `"1"` | 玩家 2 的内置 AI。敌方由地图脚本指挥，更强的 AI 可能抢指挥权、破坏计策（sc2te 默认 "7"） |
| `obs_regions` / `obs_own_position` | True | 见 4.4 |
| `obs_plan_phase` / `state_plan_phase` | False | 计划阶段 one-hot（需 StratagemDSL） |
| `region_actions` / `region_action` | True / `"move"` | 见 4.3 |
| `force_actions` | True | 见 4.3 |
| `ability_actions` | `[]` | 见 4.3 |
| `min_shoot_range` | 6.0 | 见 4.3 |
| `reward_phase` | 0.0 | 计划阶段奖励 |
| `reward_tuple` | True | True 同 sc2te（奖励元组），False 同 SMAC（标量） |
| `termination` | `"intent"` | `intent` / `plan` / `army` |
| `defeat_on_army_loss` / `failure_first` | True / True | |
| `plan_tracking` | `"auto"` | 装了 StratagemDSL 就开；`on` / `off` |
| `info_detail` | False | True 时把 `last_detail` 也放进 info |
| `engine` | `"sc2"` | `"fake"`：不启动 SC2，用 PlanSim 替身 |
| `install_map` | True | 有 SC2PATH 时把地图复制到 `$SC2PATH/Maps/Stratagem_Maps/`，否则按绝对路径加载 |

## 5. 注意事项和已知限制

* **`mission_time()` / `after(t)` 的回合语义**：地图脚本里它们是 `GameGetMissionTime()`，从**整局游戏**开始计时；MARL 地图靠刷兵
  进入下一回合，计时不归零，所以脚本里 `after(60)` 这类转移在第二回合起会立刻触发。本环境求值 intent 时按**本回合**计时。
  计划里有时间条件时，建议用 `--gameover` 打包（每回合 RequestRestartGame，计时归零）。
* **内置 AI**：PySC2/SMAC 的对手是电脑玩家。先用 `env.save_replay()` 录一局看回放，确认敌方按计划行动；如果 AI 在抢指挥，保持 `difficulty="1"`。
* **技能**：预设只覆盖常见的变形/钻地/治疗/神经寄生等。像偷梁换柱（工蜂造建筑）、反客为主（折跃门折跃）、上屋抽梯（棱镜装载）
  这类以技能为核心的计策，需要按 4.3 自定义技能动作（必要时子类化 `_ability_available`）。
* **控制权转移**（神经寄生）：单位按 tag 留在原槽位，`owner` 变化体现在原始单位数据里。
* **计划跟踪是估计**：地图脚本的真实 marking 在游戏内部，API 读不到。`plan.py` 用 StratagemDSL 的 PlanTracker 在观测上重放 Petri 网，
  并按 Galaxy 的语义**逐单位**派发进入/离开区域事件（例如 13 只单位进入诱敌区会连续推进两个转移）。地图默认关闭战争迷雾，跟踪很接近，但
  `targeting`、`elapsed` 等事实仍可能有出入。
* **fake 引擎只用于调试流程**：数值、寻路、碰撞、技能都是近似，`spawn(...)` 的援军开局就在。不能代替真机评测。
* **验证情况**：离线测试（fake 引擎）覆盖了 StratagemDSL 自带的 12 个计策例子和 2 个天梯图示例
  （导出、槽位、动作、观测/状态维度、终止、重置、gameover 重开、技能）。真机（Windows，SC2 Base98310）上用
  `dhls_18_enter_signal` 验证过：地图加载、槽位匹配、intent 胜负判定、回合间靠地图触发器重刷（`restarts: 0`）；
  随机策略 0/3，计划跟随基线 3/3 胜；WQMIX 仓库的 QMIX 能启动训练。其他计策图请先跑一遍 `quickstart.py` / `plan_follower.py`。

## 6. 常见问题

| 现象 | 原因和解决 |
| --- | --- |
| Windows 弹窗 “dot.exe - 无法找到入口：无法定位程序输入点 libintl_dgettext 于动态链接库 …\fontconfig-1.dll” | conda 里的 Graphviz 依赖库版本不一致（常见于混用 defaults 和 conda-forge 频道）。导出器默认已不调用 `dot`，更新本包即可；想修好 Graphviz：`conda install -c conda-forge --force-reinstall graphviz fontconfig gettext`，`dot -V` 不再弹窗即可。Graphviz 只用于画图，也可以直接 `conda remove graphviz` |
| `python: can't open file '...\StratagemDSL-MARL\tools\strat2sc2te.py'` | 命令里的 `../StratagemDSL-MARL` 是相对 StratagemDSL-RL 目录的路径，要求两个文件夹并排放；放在别处就改成实际路径 |
| `StratagemDSL is not installed in this Python environment` | 在装了 StratagemDSL 的环境里运行导出器（`pip install -e .`） |
| `FileNotFoundError: ... build\sc2te\<名字>` | 场景路径是相对**当前目录**的；在 StratagemDSL-RL 根目录运行，或写绝对路径（Windows 建议用正斜杠 `E:/...`） |
| `quickstart.py: unrecognized arguments` 或看不到 `map loaded` 行 | 用的是旧文件：解压时多套了一层目录。覆盖后重新 `python install.py`（环境加载的是 `site-packages/smac/env/sc2_tactics/stratagem` 里的拷贝） |
| `--idle` 时画面一动不动 | 预期行为：己方 stop、敌方等待诱敌，跑满 `episode_limit` 步后 timeout；脚本每 20 步打印一行进度 |
| `ModuleNotFoundError: No module named 'pkg_resources'`（sacred） | 新版 setuptools 去掉了 pkg_resources：`pip install "setuptools<81"` |
| PyMARL `KeyError: 'unit_dim'` | pymarl2/wqmix 一类分支要读的字段，新版 `get_env_info()` 已提供（`unit_dim`、`n_allies`、`obs_component`、`state_component`……）；更新后重新 `python install.py` |
| `reward, delta_enemy, delta_deaths, delta_ally = reward` → `cannot unpack non-iterable float` | 按 sc2te 改过的 runner 要奖励四元组：`env_args.reward_tuple=True`；原版 PyMARL 则用 `False` |
| PyMARL 一启动就去加载别的模型 / 维度不匹配 | 算法 yaml 里 `evaluate: True`、`checkpoint_path`、`load_step` 是旧实验留下的；训练时 `evaluate=False checkpoint_path="" load_step=0` |
| `load() missing 1 required positional argument: 'Loader'` | 老 PyMARL 代码配新 PyYAML：`yaml.load(f)` 改成 `yaml.load(f, Loader=yaml.FullLoader)` |

## 7. 测试

```bash
# 先导出示例（StratagemDSL-RL 根目录）
python ../StratagemDSL-MARL/tools/strat2sc2te.py examples/stratagems/*.strat examples/demos/*.strat -o build/sc2te
# 离线测试（不需要 SC2）
STRAT_SPECS=build/sc2te python -m pytest -q ../StratagemDSL-MARL/tests/test_offline.py
```

计划跟随基线在 fake 引擎上的结果（每图 2 局）可以用来判断“这张图在当前动作空间里是否可赢”：例如 `dhls`（调虎离山）、`adcc`（暗渡陈仓）
全胜；依赖技能的计策（`jctq` 钻地、`jdsr` 神经寄生、`swct` 装载、`tlhz` 造建筑等）只靠移动/攻击赢不了，是加技能动作的信号。
