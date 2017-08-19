function Tokens = ScanFirst(String, delimiter)
%used to read in first line of file, since Matlab's textscan will discard
%the last 
%inputs:
%String - string to be separated into tokens.
%delimiter - character delimiter to separate 'String' into tokens.
%outputs:
%Tokens - cell array of strings, extracted from 'String'.

%get indexes of delimiters
Delimiters = strfind(String, sprintf(delimiter));

%initialize output
Tokens = cell(1,length(Delimiters)+1);

%check number of entries
if(length(Tokens) > 1)
    
    %loop through, collecting tokens
    for i = 1:length(Tokens)-1
        
        %read strings between delimiters
        if(i == 1)
            Tokens{1} = String(1:Delimiters(i)-1);
        else
            Tokens{i} = String(Delimiters(i-1)+1:Delimiters(i)-1);
        end
        
        %trim
        Tokens{1} = strtrim(Tokens{1});
    
    end
    
    %capture last token
    Tokens{end} = String(Delimiters(end)+1:end);
    
    %trim
    Tokens{end} = strtrim(Tokens{end});
    
else
    
    %remove leading, trailing whitespace and return
    Tokens{1} = strtrim(String);
    
end
end