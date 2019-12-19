using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using CoreML;
using Falante.Interfaces;
using Foundation;

[assembly: Xamarin.Forms.Dependency(typeof(Falante.iOS.Classes.Predictor))]
namespace Falante.iOS.Classes
{
    public class Predictor : IPredictor
    {
        predictor_en model_en;
        predictor_es model_es;
        predictor_pt model_pt;

        public Predictor()
        {
        }

        public string Language { get; set; } = "en";

        public List<string> PredictNextWord(string word)
        {

            NSError err = new NSError();
            var input = new MLMultiArray(new nint[] { 1, 1 }, MLMultiArrayDataType.Int32, out err);

            Dictionary<int, String> idx2word;
            Dictionary<String, int> word2idx;

            switch (Language)
            {
                case "pt":

                    if (model_pt == null) model_pt = new predictor_pt();

                    idx2word = Newtonsoft.Json.JsonConvert.DeserializeObject<Dictionary<int, String>>(File.ReadAllText("idx2word_pt.txt"));
                    word2idx = Newtonsoft.Json.JsonConvert.DeserializeObject<Dictionary<String, int>>(File.ReadAllText("word2idx_pt.txt"));

                    if (!word2idx.ContainsKey(word)) return new List<string>();

                    input[0] = word2idx[word];

                    var output = model_pt.GetPrediction(input, out err);

                    List<Double> res = new List<double>();

                    for (var i = 0; i < idx2word.Count + 1; i++)
                    {
                        res.Add(output.Identity[new nint[] { 0, 0, i }].DoubleValue);
                    }

                    return res.OrderByDescending(x => x).Select(x2 => idx2word[res.IndexOf(x2)]).Take(10).ToList();

                case "es":

                    if (model_es == null) model_es = new predictor_es();

                    idx2word = Newtonsoft.Json.JsonConvert.DeserializeObject<Dictionary<int, String>>(File.ReadAllText("idx2word_es.txt"));
                    word2idx = Newtonsoft.Json.JsonConvert.DeserializeObject<Dictionary<String, int>>(File.ReadAllText("word2idx_es.txt"));

                    if (!word2idx.ContainsKey(word)) return new List<string>();

                    input[0] = word2idx[word];

                    var output2 = model_es.GetPrediction(input, out err);

                    List<Double> res2 = new List<double>();

                    for (var i = 0; i < idx2word.Count + 1; i++)
                    {
                        res2.Add(output2.Identity[new nint[] { 0, 0, i }].DoubleValue);
                    }

                    return res2.OrderByDescending(x => x).Select(x2 => idx2word[res2.IndexOf(x2)]).Take(10).ToList();

                default:

                    if (model_en == null) model_en = new predictor_en();


                    idx2word = Newtonsoft.Json.JsonConvert.DeserializeObject<Dictionary<int, String>>(File.ReadAllText("idx2word_en.txt"));
                    word2idx = Newtonsoft.Json.JsonConvert.DeserializeObject<Dictionary<String, int>>(File.ReadAllText("word2idx_en.txt"));

                    if (!word2idx.ContainsKey(word)) return new List<string>();

                    input[0] = word2idx[word];

                    var output3 = model_en.GetPrediction(input, out err);

                    List<Double> res3 = new List<double>();

                    for (var i = 0; i < idx2word.Count + 1; i++)
                    {
                        res3.Add(output3.Identity[new nint[] { 0, 0, i }].DoubleValue);
                    }

                    return res3.OrderByDescending(x => x).Select(x2 => idx2word[res3.IndexOf(x2)]).Take(10).ToList();

            }


        }
    }
}
