from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class LogEntry:
    """
    Entry that represents a single parsed Zookeeper log record
    """

    timestamp: str
    level: str
    thread_component: str
    message: str
    raw: str

    @property
    def primary_component(self) -> str:
        """
        Extracts root component name from ZooKeeper thread strings.

        Examples:
        - 'QuorumPeer[myid=1]/0:0:0:0:0:0:0:0:2181:FastLeaderElection@774' -> 'FastLeaderElection'
        - '/10.10.34.11:3888:QuorumCnxManager$Listener@493'               -> 'QuorumCnxManager'
        - 'SendWorker:188978561024:QuorumCnxManager$SendWorker@688'     -> 'QuorumCnxManager'
        """
        # Strip line number following '@'
        component_part = self.thread_component.split("@")[0]
        # Extract class name following final colon or slash separator
        raw_name = component_part.split(":")[-1].split("/")[-1]
        # Normalize inner class designations (e.g. QuorumCnxManager$SendWorker -> QuorumCnxManager)
        return raw_name.split("$")[0] if raw_name else self.thread_component

    @property
    def is_warning_or_error(self) -> bool:
        """
        Indicates if log level specifies issue
        """
        return self.level in ("ERROR", "WARN")


@dataclass(frozen=True)
class SFTRecord:
    """
    Represents the prompt / output pair to be used for model tuning
    """

    instruction: str
    output: str

    def to_dict(self) -> dict:
        return asdict(self)
