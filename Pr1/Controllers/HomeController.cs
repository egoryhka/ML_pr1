using Microsoft.AspNetCore.Mvc;
using Microsoft.ML.Data;
using Microsoft.ML;
using Pr1.Models;
using System.Diagnostics;

namespace Pr1.Controllers
{
    public class HomeController : Controller
    {
        private readonly ILogger<HomeController> _logger;

        public HomeController(ILogger<HomeController> logger)
        {
            _logger = logger;
        }

        [HttpGet]
        public IActionResult Index()
        {
            return View();
        }

        [HttpPost]
        public IActionResult Index(IFormFile testFile)
        {
            if (testFile == null) return View();

            var bytes = new byte[testFile.Length];
            using var stream = new MemoryStream(bytes);
            testFile.CopyTo(stream);

            MLModel.ModelInput sampleData = new MLModel.ModelInput() { ImageSource = bytes, };

            var labelBuffer = new VBuffer<ReadOnlyMemory<char>>();
            MLModel.PredictEngine.Value.OutputSchema["Score"].Annotations.GetValue("SlotNames", ref labelBuffer);
            var labels = labelBuffer.DenseValues().Select(l => l.ToString()).ToList();
            var scores = MLModel.Predict(sampleData).Score.ToList();

            var res = new List<RecogResult>();
            for (int i = 0; i < labels.Count; i++)
                res.Add(new RecogResult
                {
                    Label = labels[i],
                    Score = scores[i]
                });

            res = res.OrderByDescending(x => x.Score).ToList();
            return View(res);
        }

        public IActionResult Privacy()
        {
            return View();
        }

        [ResponseCache(Duration = 0, Location = ResponseCacheLocation.None, NoStore = true)]
        public IActionResult Error()
        {
            return View(new ErrorViewModel { RequestId = Activity.Current?.Id ?? HttpContext.TraceIdentifier });
        }
    }
}
