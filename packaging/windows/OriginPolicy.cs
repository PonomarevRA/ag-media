namespace AgMediaStudio;

/// <summary>Pure origin rules shared by navigation and native microphone consent.</summary>
internal static class OriginPolicy
{
    /// <summary>Accept only the launched server's exact HTTP loopback host and effective port, without credentials.</summary>
    public static bool IsOwned(Uri? origin, string value)
    {
        if (origin is null || !Uri.TryCreate(value, UriKind.Absolute, out var uri)) return false;
        return uri.Scheme == "http" && uri.Host == "127.0.0.1" && uri.Port == origin.Port && string.IsNullOrEmpty(uri.UserInfo);
    }

    /// <summary>Permit a consent question only for microphone capture at the owned origin; this never grants access.</summary>
    public static bool MayAskMicrophone(Uri? origin, string value, bool microphoneRequested)
        => microphoneRequested && IsOwned(origin, value);
}
