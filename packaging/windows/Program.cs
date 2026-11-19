using System.Text.Json;

namespace AgMediaStudio;

internal static class Program
{
    /// <summary>Launch a real Windows frame, or exercise owned-backend startup/shutdown for CI.</summary>
    [STAThread]
    private static int Main(string[] args)
    {
        if (args.Length == 2 && args[0] == "--backend-smoke")
        {
            return SmokeAsync(args[1]).GetAwaiter().GetResult();
        }
        ApplicationConfiguration.Initialize();
        using var context = new ApplicationContext();
        using var window = new StudioWindow();
        // The frame closes immediately. Keep its message pump alive only for bounded
        // async cleanup, so startup continuations cannot deadlock after the loop exits.
        window.FormClosed += async (_, _) =>
        {
            try { await window.ShutdownAsync(); }
            finally { context.ExitThread(); }
        };
        window.Show();
        Application.Run(context);
        return 0;
    }

    /// <summary>Test the bundled interpreter without requiring an interactive desktop or WebView2.</summary>
    private static async Task<int> SmokeAsync(string data)
    {
        try
        {
            await using var backend = new OwnedBackend(data);
            await backend.StartAsync();
            string evidence = JsonSerializer.Serialize(new { schema = "ag-media-windows-smoke-v1", origin = backend.Origin, data = backend.DataDirectory, backendReady = true });
            File.WriteAllText(Path.Combine(data, "windows-smoke.json"), evidence);
            return 0;
        }
        catch (Exception error)
        {
            Directory.CreateDirectory(data);
            File.WriteAllText(Path.Combine(data, "windows-smoke-error.txt"), error.ToString());
            return 1;
        }
    }
}
