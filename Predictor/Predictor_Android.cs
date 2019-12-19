using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using Android.App;
using Android.Content;
using Android.Content.Res;
using Falante.Interfaces;
using Java.IO;
using Java.Nio.Channels;

[assembly: Xamarin.Forms.Dependency(typeof(Falante.Android.Classes.Predictor))]
namespace Falante.Android.Classes
{
    public class Predictor : IPredictor
    {

        public static Context context;

        Xamarin.TensorFlow.Lite.Interpreter model;

        public Predictor()
        {
        }

        public string Language { get; set; } = "en";

        Dictionary<int, String> idx2word;
        Dictionary<String, int> word2idx;

        public List<string> PredictNextWord(string word)
        {

            switch (Language)
            {
                case "pt":

                    using (Stream ms = context.Assets.Open("Models/idx2word_pt.txt"))
                    {
                        using (var sr = new StreamReader(ms))
                        {
                            idx2word = Newtonsoft.Json.JsonConvert.DeserializeObject<Dictionary<int, String>>(sr.ReadToEnd());
                        }
                    }

                    using (Stream ms = context.Assets.Open("Models/word2idx_pt.txt"))
                    {
                        using (var sr = new StreamReader(ms))
                        {
                            word2idx = Newtonsoft.Json.JsonConvert.DeserializeObject<Dictionary<String, int>>(sr.ReadToEnd());
                        }
                    }

                    if (!word2idx.ContainsKey(word)) return new List<string>();

                    var il = new float[10];

                    for (var i = 0; i < 10; i++)
                    {
                        il[i] = (float)(word2idx[word]);
                    }

                    if (model == null)
                    {
                        var assets = Application.Context.Assets;
                        AssetFileDescriptor fileDescriptor = assets.OpenFd("Models/predictor_pt.tflite");
                        FileInputStream inputStream = new FileInputStream(fileDescriptor.FileDescriptor);
                        FileChannel fileChannel = inputStream.Channel;
                        long startOffset = fileDescriptor.StartOffset;
                        long declaredLength = fileDescriptor.DeclaredLength;
                        var asd = fileChannel.Map(FileChannel.MapMode.ReadOnly, startOffset, declaredLength);
                        model = new Xamarin.TensorFlow.Lite.Interpreter(asd);
                    }

                    byte[] ibytes = new byte[il.Length * sizeof(float)];
                    Buffer.BlockCopy(il, 0, ibytes, 0, ibytes.Length);

                    var bytebuffer = Java.Nio.ByteBuffer.Wrap(ibytes);
                    var output = Java.Nio.ByteBuffer.AllocateDirect(4 * 10 * (idx2word.Count + 1));

                    model.Run(bytebuffer, output);

                    List<Double> res = new List<double>();

                    var buffer = new byte[4 * 10 * (idx2word.Count + 1)];

                    Marshal.Copy(output.GetDirectBufferAddress(), buffer, 0, 4 * 10 * (idx2word.Count + 1));

                    for (var i = 0; i < idx2word.Count + 1; i++)
                    {
                        res.Add(BitConverter.ToSingle(buffer, i * 4));
                    }

                    return res.OrderByDescending(x => x).Select(x2 => idx2word[res.IndexOf(x2)]).Take(10).ToList();

                case "es":

                    using (Stream ms = context.Assets.Open("Models/idx2word_es.txt"))
                    {
                        using (var sr = new StreamReader(ms))
                        {
                            idx2word = Newtonsoft.Json.JsonConvert.DeserializeObject<Dictionary<int, String>>(sr.ReadToEnd());
                        }
                    }

                    using (Stream ms = context.Assets.Open("Models/word2idx_es.txt"))
                    {
                        using (var sr = new StreamReader(ms))
                        {
                            word2idx = Newtonsoft.Json.JsonConvert.DeserializeObject<Dictionary<String, int>>(sr.ReadToEnd());
                        }
                    }

                    if (!word2idx.ContainsKey(word)) return new List<string>();

                    var il2 = new float[10];

                    for (var i = 0; i < 10; i++)
                    {
                        il2[i] = (float)(word2idx[word]);
                    }

                    if (model == null)
                    {
                        var assets = Application.Context.Assets;
                        AssetFileDescriptor fileDescriptor = assets.OpenFd("Models/predictor_es.tflite");
                        FileInputStream inputStream = new FileInputStream(fileDescriptor.FileDescriptor);
                        FileChannel fileChannel = inputStream.Channel;
                        long startOffset = fileDescriptor.StartOffset;
                        long declaredLength = fileDescriptor.DeclaredLength;
                        var asd = fileChannel.Map(FileChannel.MapMode.ReadOnly, startOffset, declaredLength);
                        model = new Xamarin.TensorFlow.Lite.Interpreter(asd);
                    }

                    byte[] ibytes2 = new byte[il2.Length * sizeof(float)];
                    Buffer.BlockCopy(il2, 0, ibytes2, 0, ibytes2.Length);

                    var bytebuffer2 = Java.Nio.ByteBuffer.Wrap(ibytes2);
                    var output2 = Java.Nio.ByteBuffer.AllocateDirect(4 * 10 * (idx2word.Count + 1));

                    model.Run(bytebuffer2, output2);

                    List<Double> res2 = new List<double>();

                    var buffer2 = new byte[4 * 10 * (idx2word.Count + 1)];

                    Marshal.Copy(output2.GetDirectBufferAddress(), buffer2, 0, 4 * 10 * (idx2word.Count + 1));

                    for (var i = 0; i < idx2word.Count + 1; i++)
                    {
                        res2.Add(BitConverter.ToSingle(buffer2, i * 4));
                    }

                    return res2.OrderByDescending(x => x).Select(x2 => idx2word[res2.IndexOf(x2)]).Take(10).ToList();

                default:

                    using (Stream ms = context.Assets.Open("Models/idx2word_en.txt"))
                    {
                        using (var sr = new StreamReader(ms))
                        {
                            idx2word = Newtonsoft.Json.JsonConvert.DeserializeObject<Dictionary<int, String>>(sr.ReadToEnd());
                        }
                    }

                    using (Stream ms = context.Assets.Open("Models/word2idx_en.txt"))
                    {
                        using (var sr = new StreamReader(ms))
                        {
                            word2idx = Newtonsoft.Json.JsonConvert.DeserializeObject<Dictionary<String, int>>(sr.ReadToEnd());
                        }
                    }

                    if (!word2idx.ContainsKey(word)) return new List<string>();

                    var il3 = new float[10];

                    for (var i = 0; i < 10; i++)
                    {
                        il3[i] = (float)(word2idx[word]);
                        System.Diagnostics.Debug.WriteLine(il3[i]);
                    }

                    if (model == null)
                    {
                        var assets = Application.Context.Assets;
                        AssetFileDescriptor fileDescriptor = assets.OpenFd("Models/predictor_en.tflite");
                        FileInputStream inputStream = new FileInputStream(fileDescriptor.FileDescriptor);
                        FileChannel fileChannel = inputStream.Channel;
                        long startOffset = fileDescriptor.StartOffset;
                        long declaredLength = fileDescriptor.DeclaredLength;
                        var asd = fileChannel.Map(FileChannel.MapMode.ReadOnly, startOffset, declaredLength);
                        model = new Xamarin.TensorFlow.Lite.Interpreter(asd);
                    }

                    byte[] ibytes3 = new byte[il3.Length * sizeof(float)];
                    Buffer.BlockCopy(il3, 0, ibytes3, 0, ibytes3.Length);

                    var bytebuffer3 = Java.Nio.ByteBuffer.Wrap(ibytes3);
                    var output3 = Java.Nio.ByteBuffer.AllocateDirect(4 * 10 * (idx2word.Count + 1));

                    model.Run(bytebuffer3, output3);

                    List<Double> res3 = new List<double>();

                    var buffer3 = new byte[4 * 10 * (idx2word.Count + 1)];

                    Marshal.Copy(output3.GetDirectBufferAddress(), buffer3, 0, 4 * 10 * (idx2word.Count + 1));

                    for (var i = 0; i < idx2word.Count + 1; i++)
                    {
                        res3.Add(BitConverter.ToSingle(buffer3, i * 4));
                    }

                    return res3.OrderByDescending(x => x).Select(x2 => idx2word[res3.IndexOf(x2)]).Take(10).ToList();

            }

        }
    }
}
