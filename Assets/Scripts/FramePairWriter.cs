// frame图与json配对写入
using System;
using System.IO;
using System.Text;

namespace CyberLife.Vision
{
    public static class FramePairWriter
    {
        public static void Write(
            string directory,
            long frameId,
            byte[] jpeg,
            string json)
        {
            Directory.CreateDirectory(directory);

            string baseName = $"frame_{frameId:D6}";
            string basePath = Path.Combine(directory, baseName);

            string jpgPath = basePath + ".jpg";
            string jsonPath = basePath + ".json";
            string jpgTemp = jpgPath + ".tmp";
            string jsonTemp = jsonPath + ".tmp";

            if (File.Exists(jpgPath) || File.Exists(jsonPath))
            {
                throw new IOException(
                    $"重复帧文件 frame={frameId}, directory={directory}");
            }

            File.WriteAllBytes(jpgTemp, jpeg);
            File.WriteAllText(
                jsonTemp,
                json,
                new UTF8Encoding(encoderShouldEmitUTF8Identifier: false));

            // JPG 先落地，JSON 后落地。
            // JSON 文件可以视为这一帧“完整配对”的提交标志。
            File.Move(jpgTemp, jpgPath);
            File.Move(jsonTemp, jsonPath);
        }
    }
}
