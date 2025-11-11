# Examples - 示例和替代实现

## `alternative_implementations/`

包含不使用配置文件系统的替代实现方式。

### `run_plugin_nvt_0V_15ns.py`

直接在Python代码中设置所有参数的实现方式。

**优点**:
- 一个文件包含所有逻辑
- 不需要配置文件

**缺点**:
- 修改参数需要改代码
- 不如配置文件系统灵活

**推荐**: 对于大多数用户，使用配置文件系统更方便：
```bash
python3 run_from_config.py
```

## 配置文件示例

更多示例配置文件在 `../configs/` 目录。
