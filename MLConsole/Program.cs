// Initialize the Python runtime

using Python.Runtime;
using System.Diagnostics;

// Initialize the Python runtime
//ExecuteScript();
Runtime.PythonDLL = $"C:\\Python311\\python311.dll";

PythonEngine.Initialize();

using (Py.GIL())
{
    dynamic sys = Py.Import("sys");
    sys.path.append($"{Directory.GetCurrentDirectory()}\\PyModules"); // Add the parent directory of PyEnv to the system path
    //Console.WriteLine("Python interpreter path: " + sys.executable);

    dynamic njLanguagePrediction = Py.Import("NjMl"); // Import the Python module
    try
    {
        var datasetPath = "C:/Workspace/ML/Data/Codeforces/dump-original.jsonl/dump-original.jsonl";

        njLanguagePrediction.create_partial_dataframe(datasetPath);
        // njLanguagePrediction.main(); // Call the main function of the module
    }
    catch (PythonException ex)
    {
        Debug.WriteLine(ex.Message);
    }
}

//dynamic njLanguagePrediction = Py.Import("PyEnv.NjLanguagePrediction"); // Import the Python module


void ExecuteScript()
{
    PythonEngine.Initialize();

    using (Py.GIL())
    {
        // Set the path to the script
        string scriptPath = "PyEnv/LanguagePrediction.py";

        // Read the contents of the script file
        string scriptContents = File.ReadAllText(scriptPath);

        // Execute the script
        var result = PythonEngine.RunSimpleString(scriptContents);
    }
}