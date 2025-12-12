#!/bin/bash
# ═══════════════════════════════════════════════════════════
# Voltage Scan Script
#
# 自动生成并运行不同电压的模拟
#
# 使用方法:
#   ./voltage_scan.sh
# ═══════════════════════════════════════════════════════════

# 基础配置文件
BASE_CONFIG="simulation_config.ini"

# 检查基础配置文件是否存在
if [ ! -f "$BASE_CONFIG" ]; then
    echo "错误: 找不到基础配置文件: $BASE_CONFIG"
    exit 1
fi

# 要扫描的电压列表 (Volts)
VOLTAGES=(0.0 0.5 1.0 1.5 2.0)

# 模拟时间 (ns) - 可以改成更短的用于测试
SIMULATION_TIME="0.5"

echo "═══════════════════════════════════════════════════════════"
echo "Voltage Scan - 自动运行多个电压的模拟"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "基础配置: $BASE_CONFIG"
echo "电压列表: ${VOLTAGES[@]} V"
echo "模拟时间: $SIMULATION_TIME ns"
echo ""
echo "═══════════════════════════════════════════════════════════"
echo ""

# 询问用户是否继续
read -p "继续? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "取消。"
    exit 0
fi

# 循环每个电压
for voltage in "${VOLTAGES[@]}"; do
    echo ""
    echo "───────────────────────────────────────────────────────────"
    echo "运行电压: ${voltage} V"
    echo "───────────────────────────────────────────────────────────"

    # 创建临时配置文件
    TEMP_CONFIG="temp_voltage_${voltage}V.ini"

    # 复制基础配置
    cp "$BASE_CONFIG" "$TEMP_CONFIG"

    # 修改电压
    sed -i "s/^voltage = .*/voltage = $voltage/" "$TEMP_CONFIG"

    # 修改输出目录
    OUTPUT_DIR="voltage_${voltage}V_${SIMULATION_TIME}ns"
    sed -i "s|^output_dir = .*|output_dir = $OUTPUT_DIR|" "$TEMP_CONFIG"

    # 修改模拟时间（如果需要）
    sed -i "s/^total_time_ns = .*/total_time_ns = $SIMULATION_TIME/" "$TEMP_CONFIG"

    echo "配置文件: $TEMP_CONFIG"
    echo "输出目录: $OUTPUT_DIR"

    # 运行模拟
    python3 run_from_config.py "$TEMP_CONFIG"

    # 检查是否成功
    if [ $? -eq 0 ]; then
        echo "✓ 电压 ${voltage} V 完成"
        # 删除临时配置文件
        rm "$TEMP_CONFIG"
    else
        echo "✗ 电压 ${voltage} V 失败"
        echo "  保留配置文件: $TEMP_CONFIG"
        # 询问是否继续
        read -p "继续下一个电压? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "停止voltage scan。"
            exit 1
        fi
    fi
done

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "✅ Voltage Scan完成!"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "输出目录:"
for voltage in "${VOLTAGES[@]}"; do
    OUTPUT_DIR="voltage_${voltage}V_${SIMULATION_TIME}ns"
    if [ -d "$OUTPUT_DIR" ]; then
        echo "  ✓ $OUTPUT_DIR"
    else
        echo "  ✗ $OUTPUT_DIR (不存在)"
    fi
done
echo ""
