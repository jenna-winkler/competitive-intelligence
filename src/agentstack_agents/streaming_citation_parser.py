from enum import Enum
from agentstack_sdk.a2a.extensions.ui.citation import Citation


class State(Enum):
    INITIAL = "initial"
    LINK_TEXT = "link_text"
    LINK_MIDDLE = "link_middle"
    LINK_LOCATION = "link_location"
    DONE = "done"


class StreamingCitationParser:
    def __init__(self):
        self.buffer = ""
        self.state = State.INITIAL
        self.maybe_link_start = 0
        self.link_text = ""
        self.link_url = ""
        self.citations = []
        self.clean_position = 0

    def process_chunk(self, chunk: str) -> tuple[str, list[Citation]]:
        self.buffer += chunk
        output = ""
        new_citations = []

        i = self.maybe_link_start

        while i < len(self.buffer):
            char = self.buffer[i]

            if self.state == State.INITIAL:
                if char == "[":
                    if i > 0 and self.buffer[i - 1] == "!":
                        i += 1
                        continue

                    output += self.buffer[self.maybe_link_start : i]
                    self.maybe_link_start = i
                    self.link_text = ""
                    self.link_url = ""
                    self.state = State.LINK_TEXT
                    i += 1
                else:
                    i += 1

            elif self.state == State.LINK_TEXT:
                if char == "]":
                    self.state = State.LINK_MIDDLE
                    i += 1
                elif char == "\n":
                    self.state = State.INITIAL
                    self.maybe_link_start = i
                elif char == "[":
                    output += self.buffer[self.maybe_link_start : i]
                    self.maybe_link_start = i
                    self.link_text = ""
                    i += 1
                else:
                    self.link_text += char
                    i += 1

            elif self.state == State.LINK_MIDDLE:
                if char == "(":
                    self.state = State.LINK_LOCATION
                    i += 1
                else:
                    self.state = State.INITIAL
                    self.maybe_link_start = i

            elif self.state == State.LINK_LOCATION:
                if char == ")":
                    self.state = State.DONE
                    i += 1
                    break
                elif char == "\n":
                    self.state = State.INITIAL
                    self.maybe_link_start = i
                else:
                    self.link_url += char
                    i += 1

        if self.state == State.DONE:
            citation_start = self.clean_position + len(output)
            citation_end = citation_start + len(self.link_text)

            output += self.link_text

            new_citations.append(
                Citation(
                    url=self.link_url,
                    title=self.link_url.split("/")[-1].replace("-", " ").title() or self.link_text[:50],
                    description=self.link_text[:100] + ("..." if len(self.link_text) > 100 else ""),
                    start_index=citation_start,
                    end_index=citation_end,
                )
            )

            self.citations.extend(new_citations)

            self.buffer = self.buffer[i:]
            self.state = State.INITIAL
            self.maybe_link_start = 0
            self.link_text = ""
            self.link_url = ""
        else:
            if self.state == State.INITIAL:
                if i > self.maybe_link_start:
                    output += self.buffer[self.maybe_link_start : i]
                    self.buffer = self.buffer[i:]
                    self.maybe_link_start = 0
            else:
                self.buffer = self.buffer[self.maybe_link_start :]
                self.maybe_link_start = 0

        self.clean_position += len(output)

        return output, new_citations

    def finalize(self) -> str:
        output = ""

        if self.state != State.INITIAL and self.buffer:
            output = self.buffer
            self.buffer = ""
        elif self.state == State.INITIAL and self.buffer:
            output = self.buffer[self.maybe_link_start :]
            self.buffer = ""

        self.state = State.INITIAL
        self.maybe_link_start = 0

        return output

    def reset(self):
        self.buffer = ""
        self.state = State.INITIAL
        self.maybe_link_start = 0
        self.link_text = ""
        self.link_url = ""
        self.citations = []
        self.clean_position = 0
