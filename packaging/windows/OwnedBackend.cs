using System.Diagnostics;
using System.Net.Http;
using System.Text.Json;

namespace AgMediaStudio;

/// <summary>Starts the bundled interpreter and authenticates its loopback readiness record.</summary>
internal sealed class OwnedBackend : IAsyncDisposable
{
    private readonly WindowsJob job = new();
    private Process? process;
    private Task? stderrReader;
    private readonly Queue<string> errors = new();
    public Uri? Origin { get; private set; }
    public string DataDirectory { get; }

    /// <summary>Use a durable workspace outside the installation, optionally supplied by a smoke test.</summary>
    public OwnedBackend(string? data = null)
    {
        DataDirectory = Path.GetFullPath(data ?? Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "AG Media Studio", "workspace"));
        string install = Path.GetFullPath(AppContext.BaseDirectory);
        if (DataDirectory.StartsWith(install, StringComparison.OrdinalIgnoreCase))
            throw new InvalidOperationException("The workspace must be outside the installation.");
    }

    /// <summary>Wait at most 45 seconds for the child PID, nonce and HTTP endpoint to agree.</summary>
    public async Task StartAsync(CancellationToken cancellation = default)
    {
        string source = Path.Combine(AppContext.BaseDirectory, "source");
        string python = Path.Combine(AppContext.BaseDirectory, "runtime", "python", "python.exe");
        if (!File.Exists(python) || !File.Exists(Path.Combine(source, "voice_studio", "app", "desktop_runtime.py")))
            throw new FileNotFoundException("The installation is incomplete; reinstall the complete Windows package.");
        Directory.CreateDirectory(DataDirectory);
        var info = new ProcessStartInfo(python) {
            WorkingDirectory = source, UseShellExecute = false, CreateNoWindow = true,
            RedirectStandardInput = true, RedirectStandardOutput = true, RedirectStandardError = true
        };
        info.Environment.Remove("PYTHONPATH");
        info.Environment.Remove("PYTHONHOME");
        info.Environment["PYTHONUTF8"] = "1";
        // The gate closes the Process.Start-to-Job assignment race before any worker can spawn.
        string bootstrap = "import runpy,sys;sys.stdin.readline().strip()=='start' or sys.exit(71);sys.path.insert(0,sys.argv.pop(1));runpy.run_module('voice_studio.app.desktop_runtime',run_name='__main__')";
        foreach (string value in new[] { "-I", "-B", "-c", bootstrap, source, "--data", DataDirectory, "--port", "0" })
            info.ArgumentList.Add(value);
        process = Process.Start(info) ?? throw new InvalidOperationException("Could not start the bundled backend.");
        try
        {
            job.Assign(process);
            stderrReader = DrainErrorsAsync(process.StandardError);
            await process.StandardInput.WriteLineAsync("start");
            await process.StandardInput.FlushAsync();
            using var deadline = CancellationTokenSource.CreateLinkedTokenSource(cancellation);
            deadline.CancelAfter(TimeSpan.FromSeconds(45));
            while (true)
            {
                string? line = await process.StandardOutput.ReadLineAsync(deadline.Token);
                if (line is null)
                {
                    // Stdout EOF may arrive just before the diagnostic reader drains stderr.
                    await stderrReader.WaitAsync(deadline.Token);
                    throw new InvalidOperationException("The backend exited before readiness. " + ErrorSummary());
                }
                JsonDocument record;
                try { record = JsonDocument.Parse(line); }
                catch (JsonException) { continue; }
                using (record)
                {
                    JsonElement root = record.RootElement;
                    if (!root.TryGetProperty("schema", out var schema) || schema.GetString() != "ag-media-desktop-ready-v1") continue;
                    await ValidateReadinessAsync(root, deadline.Token);
                    _ = DrainOutputAsync(process.StandardOutput);
                    return;
                }
            }
        }
        catch { await DisposeAsync(); throw; }
    }

    /// <summary>Reject every readiness record that does not belong to this exact child and workspace.</summary>
    private async Task ValidateReadinessAsync(JsonElement record, CancellationToken token)
    {
        int port = record.GetProperty("port").GetInt32();
        string? nonce = record.GetProperty("instanceId").GetString();
        string expected = $"http://127.0.0.1:{port}";
        if (port is < 1 or > 65535 || record.GetProperty("pid").GetInt32() != process!.Id ||
            record.GetProperty("origin").GetString() != expected || !Guid.TryParse(nonce, out _) ||
            !string.Equals(Path.GetFullPath(record.GetProperty("data").GetString()!), DataDirectory, StringComparison.OrdinalIgnoreCase))
            throw new InvalidOperationException("Invalid owned-backend readiness record.");
        using var client = new HttpClient(new HttpClientHandler { UseProxy = false, AllowAutoRedirect = false });
        string body = await client.GetStringAsync(expected + "/api/desktop-instance", token);
        using var reply = JsonDocument.Parse(body);
        JsonElement metadata = reply.RootElement;
        if (metadata.GetProperty("instanceId").GetString() != nonce || metadata.GetProperty("pid").GetInt32() != process.Id ||
            metadata.GetProperty("origin").GetString() != expected)
            throw new InvalidOperationException("The loopback endpoint does not belong to the launched backend.");
        Origin = new Uri(expected);
    }

    /// <summary>Keep only a bounded diagnostic tail and prevent a full stderr pipe from blocking the server.</summary>
    private async Task DrainErrorsAsync(StreamReader reader)
    {
        while (await reader.ReadLineAsync() is { } line)
        {
            lock (errors) { errors.Enqueue(line.Length > 500 ? line[..500] : line); if (errors.Count > 12) errors.Dequeue(); }
        }
    }

    /// <summary>Drain post-readiness output without retaining user content.</summary>
    private static async Task DrainOutputAsync(StreamReader reader)
    {
        while (await reader.ReadLineAsync() is not null) { }
    }

    private string ErrorSummary() { lock (errors) return string.Join(Environment.NewLine, errors); }

    /// <summary>Request cooperative shutdown, wait six seconds, then close the owned job tree.</summary>
    public async ValueTask DisposeAsync()
    {
        if (process is not null)
        {
            try
            {
                if (!process.HasExited)
                {
                    await process.StandardInput.WriteLineAsync("quit");
                    await process.StandardInput.FlushAsync();
                    using var deadline = new CancellationTokenSource(TimeSpan.FromSeconds(6));
                    await process.WaitForExitAsync(deadline.Token);
                }
            }
            catch (Exception error) when (error is IOException or OperationCanceledException or InvalidOperationException) { }
            finally { job.Dispose(); process.Dispose(); process = null; }
        }
        else job.Dispose();
    }
}
