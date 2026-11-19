import Foundation
import Darwin

/// A process identity includes the kernel start time, so a reused PID is never signalled.
struct OwnedIdentity: Equatable {
    let pid: pid_t
    let startedSeconds: UInt64
    let startedMicroseconds: UInt64

    /// Read the current identity of a live process owned by this user.
    static func read(_ pid: pid_t, parent: pid_t? = nil) -> OwnedIdentity? {
        var info = proc_bsdinfo()
        let count = proc_pidinfo(pid, PROC_PIDTBSDINFO, 0, &info, Int32(MemoryLayout<proc_bsdinfo>.size))
        guard count == MemoryLayout<proc_bsdinfo>.size, info.pbi_uid == getuid() else { return nil }
        if let expected = parent, info.pbi_ppid != UInt32(expected) { return nil }
        return OwnedIdentity(pid: pid, startedSeconds: info.pbi_start_tvsec, startedMicroseconds: info.pbi_start_tvusec)
    }

    /// Signal only the exact process observed earlier; disappearance is an ordinary outcome.
    func signal(_ number: Int32) {
        guard Self.read(pid) == self else { return }
        Darwin.kill(pid, number)
    }
}

/// Track the launched service and its descendants, including workers that create another session.
final class OwnedProcesses {
    private let root: OwnedIdentity
    private var descendants: [pid_t: OwnedIdentity] = [:]

    /// Refuse ownership when the directly launched child cannot be identified by the kernel.
    init?(pid: pid_t) {
        guard let identity = OwnedIdentity.read(pid) else { return nil }
        root = identity
    }

    /// Capture descendants while their ownership is still observable, before any parent exits.
    func sample() {
        guard OwnedIdentity.read(root.pid) == root else { return }
        var pending = [root.pid]
        var visited: Set<pid_t> = []
        while let parent = pending.popLast() {
            guard visited.insert(parent).inserted else { continue }
            let parentIdentity = parent == root.pid ? root : descendants[parent]
            guard let parentIdentity = parentIdentity, OwnedIdentity.read(parent) == parentIdentity else { continue }
            // Unlike proc_listpids, this wrapper returns the PID count; its buffer size is still bytes.
            // Apple xnu/libsyscall/wrappers/libproc/libproc.c performs the byte-to-count conversion.
            let estimatedCount = proc_listchildpids(parent, nil, 0)
            guard estimatedCount > 0 else { continue }
            var children = [pid_t](repeating: 0, count: Int(estimatedCount) + 32)
            let count = children.withUnsafeMutableBytes { raw in
                proc_listchildpids(parent, raw.baseAddress, Int32(raw.count))
            }
            guard count > 0 else { continue }
            for child in children.prefix(Int(count)) where child > 0 {
                if let identity = OwnedIdentity.read(child, parent: parent), OwnedIdentity.read(parent) == parentIdentity {
                    descendants[child] = identity
                    pending.append(child)
                }
            }
        }
    }

    /// Stop descendants first, then their service, with a fixed deadline outside the AppKit thread.
    func terminate(timeout: TimeInterval = 6) {
        sample()
        for identity in descendants.values { identity.signal(SIGTERM) }
        root.signal(SIGTERM)
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            let remaining = descendants.values.contains { OwnedIdentity.read($0.pid) == $0 }
            if !remaining && OwnedIdentity.read(root.pid) != root { return }
            Thread.sleep(forTimeInterval: 0.05)
        }
        for identity in descendants.values { identity.signal(SIGKILL) }
        root.signal(SIGKILL)
    }
}
