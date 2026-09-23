# 统一数据格式

转换结果按序列保存为JSONL，每行对应一帧

```json
{
  "sequence_id": "0000",
  "frame_index": 0,
  "frame_id": "000000",
  "timestamp": 1626155123944230,
  "image_path": "image/000000.jpg",
  "pointcloud_path": "velodyne/000000.pcd",
  "objects": [
    {
      "class_name": "Car",
      "source_track_id": "001426",
      "score": 1.0,
      "box": [13.20, -3.35, -0.92, 0.08, 4.24, 1.82, 1.61]
    }
  ]
}
```

`box`顺序为`[x,y,z,yaw,length,width,height]`，坐标系为路侧虚拟激光雷达坐标系

跟踪结果保持相同帧结构，对象字段使用`track_id`表示算法输出轨迹编号
